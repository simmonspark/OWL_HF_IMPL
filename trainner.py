from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        import torch.nn.functional as F
        loss = F.mse_loss(logits, labels.float())
        return (loss, outputs) if return_outputs else loss
#dd