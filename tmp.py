
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 1. 이미지 처리 함수
def load_image(image_path):
    """ 이미지를 불러와 PIL 이미지 객체로 반환 """
    return Image.open(image_path).convert("RGB")

# 2. 텍스트 처리 함수 (Hugging Face 방식 적용)
def prepare_text_queries(objects):
    """ 탐지할 객체 리스트를 모델 입력 형식으로 변환 (문장 형태) """
    return [[f"a photo of a {obj}" for obj in objects]]

# 3. 모델 실행 및 객체 탐지 (Hugging Face 방식 적용)
def detect_objects(image, text_queries, model, processor, threshold=0.1):
    """ 이미지와 텍스트 쿼리를 입력받아 객체 탐지 수행 """
    inputs = processor(text=text_queries, images=image, return_tensors="pt")

    # 모델 실행
    with torch.no_grad():
        outputs = model(input_ids = inputs["input_ids"].cuda(), attention_mask = inputs["attention_mask"].cuda(),pixel_values = inputs["pixel_values"].cuda())

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])  # (H, W) 형태

    # 바운딩 박스 및 클래스 스코어 후처리
    results = processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

    detected_objects = []
    i = 0  # 첫 번째 이미지만 처리
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    for box, score, label in zip(boxes, scores, labels):
        box = [round(coord, 2) for coord in box.tolist()]  # 좌표값 반올림
        detected_objects.append((box, score.item(), text_queries[i][label]))

    return detected_objects

# 4. 시각화 함수
def visualize_detections(image, detections):
    """ 감지된 객체를 이미지 위에 시각화 """
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    for box, score, label in detections:
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1, f"{label}: {round(score, 3)}", bbox=dict(facecolor="yellow", alpha=0.5))

    plt.show()

# 5. 전체 실행 (메인 코드)
if __name__ == "__main__":
    import torch
    from transformers import AutoProcessor, OwlViTConfig
    from transformers import OwlViTForObjectDetection as HFMDOEL
    from models.owl_v1 import OwlViTForObjectDetection  # 사용자 정의 모델 로드

    # 1. 모델과 프로세서 로드
    MODEL_NAME = "google/owlvit-base-patch32"

    # 프로세서 로드 (Hugging Face)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # 전체 config 가져오기 및 변환
    config_dict = OwlViTConfig.get_config_dict(MODEL_NAME)
    config = OwlViTConfig(config_dict[0])
    print("Loaded Config:", config)

    # 2. 모델 로드 및 가중치 적용
    hf_model = HFMDOEL.from_pretrained(MODEL_NAME).cuda()
    state_hf = hf_model.state_dict()
    print(state_hf.keys())
    # 사용자 정의 모델 인스턴스화 및 가중치 적용
    model = OwlViTForObjectDetection(config).cuda()
    print(model.state_dict().keys())
    model.load_state_dict(state_hf)  # ✅ 반환값을 사용하지 않음

    # 3. 이미지와 탐지할 객체 설정
    image_path = "/home/sien/사진/스크린샷/스크린샷 2025-02-04 18-56-15.png"
    objects_to_detect = ["cat", "dog", "person"]  # 감지할 객체 리스트

    # 4. 이미지 및 텍스트 처리
    image = load_image(image_path)
    text_queries = prepare_text_queries(objects_to_detect)

    # 5. 객체 탐지 수행
    detections = detect_objects(image, text_queries, model, processor)

    # 6. 탐지 결과 시각화
    visualize_detections(image, detections)

