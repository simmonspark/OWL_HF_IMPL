import json
import os
import shutil

import torch
import yaml
from torch.onnx.symbolic_opset9 import unsqueeze
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.io import write_png
from tqdm import tqdm

from loss import PushPullLoss
from dataset import get_dataloaders
from models.testmodel import PostProcess
from models.final_model import load_model
from utils.train_util import (
    coco_to_model_input,
    labels_to_classnames,
    model_output_to_image,
    update_metrics,
)
from utils.util import BoxUtil, GeneralLossAccumulator, ProgressFormatter
import torch
import torchvision.ops as ops


def nms_with_classes(pre_boxes, pred_classes_with_names, iou_threshold=0.5):
    """
    NMS를 적용하면서 각 바운딩 박스와 클래스 정보를 유지하는 함수

    Args:
        pre_boxes (Tensor): [N, 4] 형태의 바운딩 박스 좌표 (x1, y1, x2, y2)
        pred_classes_with_names (list): 각 박스에 해당하는 클래스 이름 리스트
        iou_threshold (float): IoU 임계값

    Returns:
        kept_boxes (Tensor): NMS 적용 후 남은 바운딩 박스
        kept_classes (list): NMS 적용 후 남은 클래스 목록
    """

    # NMS 적용
    keep_indices = ops.nms(pre_boxes, torch.ones(len(pre_boxes)), iou_threshold)

    # NMS 적용 후 남은 요소 선택
    kept_boxes = pre_boxes[keep_indices]
    kept_classes = [pred_classes_with_names[i] for i in keep_indices]

    return kept_boxes, kept_classes


def get_training_config():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        return data["training"]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(device)
    scaler = torch.cuda.amp.GradScaler()
    general_loss = GeneralLossAccumulator()
    progress_summary = ProgressFormatter()

    if os.path.exists("debug"):
        shutil.rmtree("debug")

    training_cfg = get_training_config()
    train_dataloader, test_dataloader, scales, labelmap = get_dataloaders()

    model = load_model(labelmap, device)

    postprocess = PostProcess(
        confidence_threshold=training_cfg["confidence_threshold"],
        iou_threshold=training_cfg["iou_threshold"],
    )

    criterion = PushPullLoss(
        len(labelmap),
        scales=torch.tensor(scales).to(device)
        if training_cfg["use_class_weight"]
        else None,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=training_cfg["weight_decay"],
    )

    model.train()
    classMAPs = {v: [] for v in list(labelmap.values())}
    for epoch in range(training_cfg["n_epochs"]):
        if training_cfg["save_eval_images"]:
            os.makedirs(f"debug/{epoch}", exist_ok=True)

        # Train loop
        losses = []
        for i, (image, labels, boxes, metadata) in enumerate(
            tqdm(train_dataloader, ncols=60)
            # train_dataloader
        ):
            optimizer.zero_grad()

            # Prep inputs
            image = image.to(device)
            labels = labels.to(device)
            boxes = coco_to_model_input(boxes, metadata).to(device)

            # Predict
            all_pred_boxes, pred_classes, pred_sims, _ = model(image)
            losses = criterion(pred_sims.cuda(), labels.cuda(), all_pred_boxes.cuda(), boxes.cuda())
            loss = (
                losses["loss_ce"]
                + losses["loss_bg"]
                + losses["loss_bbox"]
                + losses["loss_giou"]
            )
            loss.backward()
            optimizer.step()

            general_loss.update(losses)

        train_metrics = general_loss.get_values()
        general_loss.reset()

        # Eval loop
        model.eval()
        with torch.no_grad():
            for i, (image, labels, boxes, metadata) in enumerate(
                tqdm(test_dataloader, ncols=60)
            ):
                # Prep inputs
                image = image.to(device)
                labels = labels.to(device)
                boxes = coco_to_model_input(boxes, metadata).to(device)

                # Get predictions and save output
                pred_boxes, pred_classes, pred_class_sims, _ = model(image)
                pred_boxes, pred_classes, scores = postprocess(
                    pred_boxes, pred_class_sims
                )

                # Use only the top 200 boxes to stay consistent with benchmarking
                top = torch.topk(scores, min(200, scores.size(-1)))
                scores = top.values
                inds = top.indices.squeeze(0)

                update_metrics(
                    metric,
                    metadata,
                    pred_boxes[:, inds],
                    pred_classes[:, inds],
                    scores,
                    boxes,
                    labels,
                )

                if training_cfg["save_eval_images"]:
                    pred_classes_with_names = labels_to_classnames(
                        pred_classes, labelmap
                    )
                    pred_boxes = model_output_to_image(pred_boxes.cpu(), metadata)
                    image_with_boxes = BoxUtil.draw_box_on_image(
                        metadata["impath"].pop(),
                        pred_boxes,
                        pred_classes_with_names,
                    )

                    write_png(image_with_boxes, f"debug/{epoch}/{i}.jpg")

        print("Computing metrics...")
        val_metrics = metric.compute()
        for i, p in enumerate(val_metrics["map_per_class"].tolist()):
            label = labelmap[str(i)]
            classMAPs[label].append(p)

        with open("class_maps.json", "w") as f:
            json.dump(classMAPs, f)

        metric.reset()
        progress_summary.update(epoch, train_metrics, val_metrics)
        progress_summary.print()
