from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 모델의 기본 출력을 가져오기
        outputs = model(**inputs)
        logits = outputs.logits  # 모델이 예측한 값 (보통 softmax 이전 값)

        # 실제 정답 라벨
        labels = inputs.get("labels")

        # 원하는 손실 함수 적용 (예: MSE Loss)
        import torch.nn.functional as F
        loss = F.mse_loss(logits, labels.float())  # 예제: MSE 손실 적용

        return (loss, outputs) if return_outputs else loss
