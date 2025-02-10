from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
import torch.nn as nn
import math
from transformers.models.owlvit.modeling_owlvit import OwlViTObjectDetectionOutput
from typing import Any, Dict, Optional, Tuple, Union

'''
vision vit:
    encoder를 돌려서 
    1. img embed : B T C
    2. pool : B C
text vision vit:
    1. text embed : B T C
    2. pool : B C
이렇게 리턴함 
'''


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
        # Create grid coordinates using torch
        x_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        y_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x_coordinates, y_coordinates, indexing="xy")

        # Stack the coordinates and divide by num_patches
        box_coordinates = torch.stack((xx, yy), dim=-1)
        box_coordinates /= num_patches

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.view(-1, 2)

        return box_coordinates

    def compute_box_bias(self, num_patches: int, feature_map=None) -> torch.Tensor:
        if feature_map is not None:
            raise ValueError("feature_map has been deprecated as an input. Please pass in num_patches instead")
        # The box center is biased to its position on the feature grid
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
        #1,576,4
        return pred_boxes

    def class_predictor(
            self,
            image_feats: torch.FloatTensor,
            query_embeds=None,
            query_mask=None,
    ):
        #idx[0] : 1, 586,3
        #idx[1] : 1, 576, 512
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds, query_mask)
        #idx[0] : 1, 586,3
        #idx[1] : 1, 576, 512
        return (pred_logits, image_class_embeds)

    def image_text_embedder(
            self,
            input_ids: torch.Tensor,
            pixel_values: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor]:

        #logits_per_image : idx[0] : 1,3
        #logits_per_text : idx[1] : 3,1
        #text_embeds : idx[2] : 3,512
        #image_embeds : idx[3] : 1,512
        #text_model_output : idx[4] :
            #idx[0] : last_hidden_state -> shape(3,16,512)
            #idx[1] : pooled_output -> shape(3,512)
        #vision_model_output : idx[5] :
            # idx[0] : last_hidden_state : 1, 577, 768
            # idx[1] : pooler_output : 1,768
        outputs = self.owlvit(
            pixel_values=pixel_values,
            input_ids=input_ids
        )
        image_embeds = self.owlvit.vision_model.post_layernorm(outputs[5][0])

        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], image_embeds[:, :-1].shape)

        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        new_size = (
            image_embeds.shape[0],
            self.sqrt_num_patches,
            self.sqrt_num_patches,
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)
        text_embeds = outputs[2]
        #idx[0] : 3,512
        #idx[1] : 1,24,24, 768
        #idx[2] :
            # logits_per_image : idx[0] : 1,3
            # logits_per_text : idx[1] : 3,1
            # text_embeds : idx[2] : 3,512
            # image_embeds : idx[3] : 1,512
            # text_model_output : idx[4] :
                # idx[0] : last_hidden_state -> shape(3,16,512)
                # idx[1] : pooled_output -> shape(3,512)
            # vision_model_output : idx[5] :
                # idx[0] : last_hidden_state : 1, 577, 768
                # idx[1] : pooler_output : 1,768
        return (text_embeds, image_embeds, outputs)



    def forward(
            self,
            input_ids,
            pixel_values
    ):
        #idx[0] : 3,512
        #idx[1] : 1,24,24, 768
        #idx[2] :
            # logits_per_image : idx[0] : 1,3
            # logits_per_text : idx[1] : 3,1
            # text_embeds : idx[2] : 3,512
            # image_embeds : idx[3] : 1,512
            # text_model_output : idx[4] :
                # idx[0] : last_hidden_state -> shape(3,16,512)
                # idx[1] : pooled_output -> shape(3,512)
            # vision_model_output : idx[5] :
                # idx[0] : last_hidden_state : 1, 577, 768
                # idx[1] : pooler_output : 1,768
        query_embeds, feature_map, outputs = self.image_text_embedder(
            input_ids=input_ids,
            pixel_values=pixel_values
        )
        # 이건 안씀
        text_outputs = outputs[4]
        vision_outputs = outputs[5]

        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))

        max_text_queries = input_ids.shape[0] // batch_size
        query_embeds = query_embeds.reshape(batch_size, max_text_queries, query_embeds.shape[-1])
        input_ids = input_ids.reshape(batch_size, max_text_queries, input_ids.shape[-1])
        query_mask = input_ids[..., 0] > 0
        #idx[0] : 1, 586,3
        #idx[1] : 1, 576, 512
        (pred_logits, class_embeds) = self.class_predictor(image_feats, query_embeds, query_mask)
        pred_boxes = self.box_predictor(image_feats, feature_map)
        # idx[0] : image_embeds -> shape(1, 24, 24, 768)
        # idx[1] : text_embeds -> shape(1,3,512)
        # idx[2] : pred_boxes -> shape(1,576,4)
        # idx[3] : logits -> shape(1,576,3)
        # idx[4] : class_embeds -> shape(1,576,512)
        # idx[5] : text_model_output
            # idx[0] : last_hidden_state -> shape(3,16,512)
            # idx[1] : pooled_output -> shape(3,512)
        # idx[6] : vision_model_output
            # idx[0] : last_hidden_state : (1, 577, 768)
            # idx[1] : pooler_output : (1,768)
        return OwlViTObjectDetectionOutput(
            image_embeds=feature_map,
            text_embeds=query_embeds,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            class_embeds=class_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class OwlViTClassPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig):
        super().__init__()

        out_dim = config.text_config.hidden_size
        self.query_dim = config.vision_config.hidden_size

        self.dense0 = nn.Linear(self.query_dim, out_dim)
        self.logit_shift = nn.Linear(self.query_dim, 1)
        self.logit_scale = nn.Linear(self.query_dim, 1)
        self.elu = nn.ELU()

    def forward(
            self,
            image_embeds: torch.FloatTensor,
            query_embeds: Optional[torch.FloatTensor],
            query_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.FloatTensor]:
        image_class_embeds = self.dense0(image_embeds)
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale
        #idx[0] : 1, 586,3
        #idx[1] : 1, 576, 512
        return (pred_logits, image_class_embeds)


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

    def get_text_features(self,input_ids=None) -> torch.FloatTensor:
        # idx[0] : last_hidden_state -> shape(3,16,512)
        # idx[1] : pooled_output -> shape(3,512)
        text_output = self.text_model(input_ids=input_ids)
        pooled_output = text_output[1]
        text_features = self.text_projection(pooled_output)
        return text_features

    def get_image_features(
            self,
            pixel_values=None
    ) -> torch.FloatTensor:

        # idx[0] : last_hidden_state : 1, 577, 768
        # idx[1] : pooler_output : 1,768
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs[1]
        image_features = self.visual_projection(pooled_output)
        return image_features

    def forward(self,input_ids=None,pixel_values=None):
        # idx[0] : last_hidden_state : 1, 577, 768
        # idx[1] : pooler_output : 1,768
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        # idx[0] : last_hidden_state -> shape(3,16,512)
        # idx[1] : pooled_output -> shape(3,512)
        text_outputs = self.text_model(input_ids=input_ids)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        image_embeds = image_embeds / torch.linalg.norm(image_embeds, ord=2, dim=-1, keepdim=True)
        text_embeds_norm = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp().to(image_embeds.device)

        logits_per_text = torch.matmul(text_embeds_norm, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        text_embeds = text_embeds_norm

        #return
        #logits_per_image : idx[0] : 1,3
        #logits_per_text : idx[1] : 3,1
        #text_embeds : idx[2] : 3,512
        #image_embeds : idx[3] : 1,512
        #text_model_output : idx[4] :
            #idx[0] : last_hidden_state -> shape(3,16,512)
            #idx[1] : pooled_output -> shape(3,512)
        #vision_model_output : idx[5] :
            # idx[0] : last_hidden_state : 1, 577, 768
            # idx[1] : pooler_output : 1,768
        return (logits_per_image,logits_per_text,text_embeds,image_embeds,text_outputs,vision_outputs)


class OwlViTVisionTransformer(nn.Module):
    def __init__(self, config):
        super(OwlViTVisionTransformer, self).__init__()
        self.config = config
        self.embeddings = OwlViTVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = OwlViTEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self,pixel_values: torch.FloatTensor):

        expected_input_dtype = self.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.to(expected_input_dtype)

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = encoder_outputs
        pooled_output = last_hidden_state[:, 0, :]

        pooled_output = self.post_layernorm(pooled_output)
        #return
        # idx[0] : last_hidden_state : 1, 577, 768
        # idx[1] : pooler_output : 1,768
        return (last_hidden_state,pooled_output)


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

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [batch_size, num_channels, height, width]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class OwlViTTextTransformer(nn.Module):
    def __init__(self, config):
        super(OwlViTTextTransformer, self).__init__()
        self.config = config.text_config
        embed_dim = config.hidden_size
        self.embeddings = OwlViTTextEmbeddings(config)
        self.encoder = OwlViTEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids=None
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states
        )
        last_hidden_state = encoder_outputs
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]
        #return
        # idx[0] : last_hidden_state -> shape(3,16,512)
        # idx[1] : pooled_output -> shape(3,512)
        return (last_hidden_state,pooled_output)


class OwlViTTextEmbeddings(nn.Module):
    def __init__(self, config):
        super(OwlViTTextEmbeddings, self).__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            inputs_embeds=None,
    ) -> torch.Tensor:

        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class OwlViTEncoder(nn.Module):
    def __init__(self, config):
        super(OwlViTEncoder, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([OwlViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,inputs_embeds):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs[0]
        return (hidden_states)


class OwlViTEncoderLayer(nn.Module):
    def __init__(self, config):
        super(OwlViTEncoderLayer, self).__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OwlViTAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = OwlViTMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self,hidden_states: torch.Tensor):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        return outputs


# o
class OwlViTAttention(nn.Module):
    def __init__(self, config):
        super(OwlViTAttention, self).__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, input: torch.Tensor, T: int, B: int):
        return input.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                hidden_states: torch.Tensor):
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = attn_probs.to(value_states.dtype)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


# o
class OwlViTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = NewGELUActivation()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# o
class NewGELUActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def owlvit_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


if __name__ == "__main__":
    from transformers import OwlViTForObjectDetection as hf_model
    from transformers import OwlViTConfig

    # 전체 config 가져오기
    config_dict = OwlViTConfig.get_config_dict("google/owlvit-base-patch32")

    # 딕셔너리를 OwlViTConfig 클래스로 변환
    config = OwlViTConfig(config_dict[0])

    # 모델 로드
    tmp = hf_model.from_pretrained("google/owlvit-base-patch32")
    state_hf = tmp.state_dict()
    print(state_hf.keys())
    model = OwlViTForObjectDetection(config)
    print(model.state_dict().keys())
