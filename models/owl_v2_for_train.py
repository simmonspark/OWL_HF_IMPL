from transformers import OwlViTProcessor
import torch
import torch.nn as nn
import math
from torchvision.ops import box_convert, generalized_box_iou
from transformers.models.owlvit.modeling_owlvit import OwlViTObjectDetectionOutput

class OwlViTConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def set(self, key, value):
        setattr(self, key, value)


class Preprocessor:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    def __call__(self, text, images, **kwargs):  # 키워드 통일
        '''
        input_ids : tokenized
        attention_mask : tokenized
        pixel_values 1, 3, 768, 768
        '''
        return self.processor(text=text, images=images, **kwargs)


def _upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes):
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


def center_to_corners_format(bboxes_center):
    return _center_to_corners_format_torch(bboxes_center)


def _center_to_corners_format_torch(bboxes_center: "torch.Tensor") -> "torch.Tensor":
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners


def prepare_text_queries(objects):
    return [objects]


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->owlvit
def owlvit_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

class GELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.activation_fn = GELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        # conf
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        # layer
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, input: torch.Tensor, T: int, B: int):
        return input.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                input: torch.Tensor,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False):
        bsz, tgt_len, embed_dim = input.size()
        query_states = self.q_proj(input) * self.scale
        key_states = self._shape(self.k_proj(input), -1, bsz)
        value_states = self._shape(self.v_proj(input), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = attn_probs.to(value_states.dtype)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class OwlViTEncoderLayer(nn.Module):
    def __init__(self, config):
        super(OwlViTEncoderLayer, self).__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, inputs_embeds, attention_mask=None):
        residual = inputs_embeds
        hidden = residual + self.self_attn(input=self.layer_norm1(inputs_embeds), attention_mask=attention_mask)
        hidden = self.layer_norm2(hidden + residual)

        residual = hidden
        hidden = residual + self.mlp(self.layer_norm2(hidden))
        return hidden


class OwlViTEncoder(nn.Module):
    def __init__(self, config):
        super(OwlViTEncoder, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([OwlViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, attention_mask=None):
        for encoder_layer in self.layers:
            inputs_embeds = encoder_layer(
                inputs_embeds, attention_mask=attention_mask,
            )
        return inputs_embeds


class OwlViTTextEmbeddings(nn.Module):
    def __init__(self, config):
        super(OwlViTTextEmbeddings, self).__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids) -> torch.Tensor:
        B, T = input_ids.shape
        position_ids = self.position_ids[:, :T]
        position_embedding = self.position_embedding(position_ids)
        token_embedding = self.token_embedding(input_ids)
        return token_embedding + position_embedding


class OwlViTTextTransformer(nn.Module):
    def __init__(self, config):
        super(OwlViTTextTransformer, self).__init__()
        self.config = config
        self.embeddings = OwlViTTextEmbeddings(config)
        self.encoder = OwlViTEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids, attention_mask = None) -> torch.Tensor:
        B, T = input_ids.shape
        hidden_states = self.embeddings(input_ids=input_ids)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=attention_mask)
        last_hidden_state = self.final_layer_norm(encoder_outputs)
        # 가장 마지막 패딩이 아닌 토큰, CLIP learning에 사용
        pool = last_hidden_state[torch.arange(B), input_ids.to(torch.int).argmax(dim=-1)]
        return dict(last_hidden_state=last_hidden_state, pooled_output=pool)


class OwlViTVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, input) -> torch.Tensor:
        batch_size = input.shape[0]
        patch_embeds = self.patch_embedding(input)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # B T C

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class OwlViTVisionTransformer(nn.Module):
    def __init__(self, config):
        super(OwlViTVisionTransformer, self).__init__()
        self.config = config
        self.embeddings = OwlViTVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = OwlViTEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        expected_input_dtype = self.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.to(expected_input_dtype)
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = encoder_outputs
        pooled_output = last_hidden_state[:, 0, :]  # 가장 첫 번쨰 토큰이 pool, CLIP learning에 사용
        pooled_output = self.post_layernorm(pooled_output)

        return dict(last_hidden_state=last_hidden_state, pooled_output=pooled_output)


class OwlViTModel(nn.Module):
    def __init__(self, config):
        super(OwlViTModel, self).__init__()
        self.config = config
        text_config = config.text_config
        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = OwlViTTextTransformer(text_config)
        self.vision_model = OwlViTVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

    def get_image_features(self, input):
        # pool 추출
        vision_outputs = self.vision_model(pixel_values=input)
        return self.visual_projection(vision_outputs['pooled_output'])

    def get_text_features(self, input, attention_mask = None):
        # pool 추출
        vision_outputs = self.text_model(pixel_values=input, attention_mask=attention_mask)
        return self.text_projection(vision_outputs['pooled_output'])

    def forward(self, input_ids, pixel_values, attention_mask = None):
        img_out = self.vision_model(pixel_values=pixel_values)
        text_out = self.text_model(input_ids=input_ids, attention_mask = attention_mask)

        text_embeds = text_out['pooled_output']
        text_embeds = self.text_projection(text_embeds)
        image_embeds = img_out['pooled_output']
        image_embeds = self.visual_projection(image_embeds)

        image_embeds = image_embeds / torch.linalg.norm(image_embeds, ord=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)

        # 이거 logit scale 역할을 잘 모르겠음 scailing factor 같은데 왜 이렇게 어렵게 쓰는거지
        logit_scale = self.logit_scale.exp().to(image_embeds.device)

        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        return dict(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_outputs=text_out,
            vision_model_output=img_out
        )


class OwlViTBoxPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig, out_dim: int = 4):
        super().__init__()

        width = config.vision_config.hidden_size
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(width, out_dim)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        output = self.dense0(image_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.dense2(output)
        return output


class OwlViTClassPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig):
        super().__init__()

        out_dim = config.text_config.hidden_size
        self.query_dim = config.vision_config.hidden_size

        self.dense0 = nn.Linear(self.query_dim, out_dim)
        self.logit_shift = nn.Linear(self.query_dim, 1)
        self.logit_scale = nn.Linear(self.query_dim, 1)
        self.elu = nn.ELU()

    def forward(self, image_embeds, query_embeds):
        image_class_embeds = self.dense0(image_embeds)
        if query_embeds is None:
            device = image_class_embeds.device
            batch_size, num_patches = image_class_embeds.shape[:2]
            pred_logits = torch.zeros((batch_size, num_patches, self.query_dim)).to(device)
            return (pred_logits, image_class_embeds)

        # Normalize image and text features
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        # Get class predictions
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        # Apply a learnable shift and scale to logits
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        return (pred_logits, image_class_embeds)


class OwlViTForObjectDetection(nn.Module):
    def __init__(self, config):
        super(OwlViTForObjectDetection, self).__init__()
        self.config = config
        self.owlvit = OwlViTModel(config)
        self.class_head = OwlViTClassPredictionHead(config)
        self.box_head = OwlViTBoxPredictionHead(config)

        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)
        self.sigmoid = nn.Sigmoid()

        self.sqrt_num_patches = config.vision_config.image_size // config.vision_config.patch_size
        self.box_bias = self.compute_box_bias(self.sqrt_num_patches)

    @staticmethod
    def normalize_grid_corner_coordinates(num_patches: int) -> torch.Tensor:
        x_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        y_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x_coordinates, y_coordinates, indexing="xy")

        box_coordinates = torch.stack((xx, yy), dim=-1)
        box_coordinates /= num_patches

        box_coordinates = box_coordinates.view(-1, 2)

        return box_coordinates

    def compute_box_bias(self, num_patches: int, feature_map=None) -> torch.Tensor:
        if feature_map is not None:
            raise ValueError("feature_map has been deprecated as an input. Please pass in num_patches instead")
        box_coordinates = self.normalize_grid_corner_coordinates(num_patches)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # The box size is biased to the patch size
        box_size = torch.full_like(box_coord_bias, 1.0 / num_patches)
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    def box_predictor(
            self,
            image_feats: torch.FloatTensor,
            feature_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                Features extracted from the image, returned by the `image_text_embedder` method.
            feature_map:
                A spatial re-arrangement of image_features, also returned by the `image_text_embedder` method.
        Returns:
            pred_boxes:
                List of predicted boxes (cxcywh normalized to 0, 1) nested within a dictionary.
        """
        # Bounding box detection head [batch_size, num_boxes, 4].
        pred_boxes = self.box_head(image_feats)

        # Compute the location of each token on the grid and use it to compute a bias for the bbox prediction
        box_bias = self.box_bias.to(feature_map.device)
        pred_boxes += box_bias
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    def class_predictor(self, image_feats: torch.FloatTensor, query_embeds=None):
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds)
        return (pred_logits, image_class_embeds)

    def image_text_embedder(self, input_ids: torch.Tensor, pixel_values: torch.FloatTensor):
        outputs = self.owlvit(pixel_values=pixel_values, input_ids=input_ids)

        last_hidden_state = outputs['vision_model_output']['last_hidden_state']
        image_embeds = self.owlvit.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], image_embeds[:, :-1].shape)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            self.sqrt_num_patches,
            self.sqrt_num_patches,
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)
        text_embeds = outputs['text_embeds']

        return (text_embeds, image_embeds, outputs)

    def image_embedder(self, pixel_values):
        vision_outputs = self.owlvit.vision_model(pixel_values=pixel_values)

        last_hidden_state = vision_outputs['last_hidden_state']
        image_embeds = self.owlvit.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], image_embeds[:, :-1].shape)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            self.sqrt_num_patches,
            self.sqrt_num_patches,
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)

        return (image_embeds, vision_outputs)

    def forward(self, input_ids, pixel_values, attention_mask = None):
        query_embeds, feature_map, outputs = self.image_text_embedder(
            input_ids=input_ids,
            pixel_values=pixel_values
        )

        # Text and vision model outputs
        text_outputs = outputs['text_model_outputs']
        vision_outputs = outputs['vision_model_output']

        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))

        # Reshape from [batch_size * max_text_queries, hidden_dim] -> [batch_size, max_text_queries, hidden_dim]
        max_text_queries = input_ids.shape[0] // batch_size
        query_embeds = query_embeds.reshape(batch_size, max_text_queries, query_embeds.shape[-1])

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        input_ids = input_ids.reshape(batch_size, max_text_queries, input_ids.shape[-1])
        query_mask = input_ids[..., 0] > 0

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.class_predictor(image_feats, query_embeds)

        # Predict object boxes
        pred_boxes = self.box_predictor(image_feats, feature_map)

        return OwlViTObjectDetectionOutput(
            image_embeds=feature_map,
            text_embeds=query_embeds,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            class_embeds=class_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs
            )
