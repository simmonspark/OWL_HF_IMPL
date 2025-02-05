import cv2
import torch
import numpy as np
from transformers import AutoProcessor, OwlViTConfig
from transformers import OwlViTForObjectDetection as HFMDOEL
from models.owl_v1 import OwlViTForObjectDetection  # 사용자 정의 모델 로드
from PIL import Image, ImageDraw, ImageFont

# 1. 모델 및 프로세서 로드
MODEL_NAME = "google/owlvit-base-patch32"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
config_dict = OwlViTConfig.get_config_dict(MODEL_NAME)
config = OwlViTConfig(config_dict[0])

hf_model = HFMDOEL.from_pretrained(MODEL_NAME).cuda()
state_hf = hf_model.state_dict()

model = OwlViTForObjectDetection(config).cuda()
model.load_state_dict(state_hf)


# 2. 객체 감지 함수
def detect_objects(image, text_queries, model, processor, threshold=0.1):
    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
        )
    target_sizes = torch.Tensor([image.size[::-1]])  # (H, W) 형태
    results = processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

    detected_objects = []
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
    for box, score, label in zip(boxes, scores, labels):
        if score.item() >= threshold:
            box = [round(coord, 2) for coord in box.tolist()]
            detected_objects.append((box, score.item(), text_queries[0][label]))
    return detected_objects


# 3. 프레임 시각화 함수
def draw_detections(frame, detections):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    for box, score, label in detections:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{label}: {round(score, 2)}", fill="yellow")
    return np.array(image)


# 4. 비디오 처리 함수
def process_video(input_path, output_path, objects_to_detect):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    text_queries = [[f"a photo of a {obj}" for obj in objects_to_detect]]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = detect_objects(image, text_queries, hf_model, processor)
        result_frame = draw_detections(frame, detections)
        out.write(cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# 5. 실행
def main():
    input_video = "/home/sien/다운로드/Video 4.mp4"  # 입력 동영상 경로
    output_video = "/home/sien/다운로드/detect.mp4"  # 출력 동영상 경로
    objects_to_detect = ["cat", "dog", "person"]  # 감지할 객체 리스트
    process_video(input_video, output_video, objects_to_detect)


if __name__ == "__main__":
    main()
