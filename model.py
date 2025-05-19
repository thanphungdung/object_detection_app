from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F
from ensemble_boxes import weighted_boxes_fusion
import os

class EnsembleDetector:
    def __init__(self, score_thresh=0.3, iou_thresh=0.5):

        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True).eval()
        self.ssd = ssdlite320_mobilenet_v3_large(pretrained=True).eval()
        self.names = self.yolo.names
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh

    def run_yolo(self,image, width, height):
        results = self.yolo(image)
        preds = results.pred[0].cpu().numpy() 
        
        boxes, scores, labels = [], [], []

        for *box, conf, cls in preds:
            x1, y1, x2, y2 = box
            boxes.append([x1 / width, y1 / height, x2 / width, y2 / height])
            scores.append(conf)
            labels.append(int(cls))
            
        return boxes, scores, labels

    def run_ssd(self, image, w, h):
        image = image.resize((320, 320))
        preds = self.ssd([F.to_tensor(image)])[0]
        boxes, scores, labels = [], [], []
        for s, l, b in zip(preds['scores'], preds['labels'], preds['boxes']):
            cid = int(l.item()) - 1
            if s > self.score_thresh and cid >= 0:
                x1, y1, x2, y2 = b.tolist()
                boxes.append([x1/w, y1/h, x2/w, y2/h])
                scores.append(s.item())
                labels.append(cid)
        return boxes, scores, labels
    def compute_iou(self,box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area else 0
    def suppress_conflicting_classes(self,boxes, scores, labels, iou_thresh=0.7):
        keep = [True] * len(boxes)

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if labels[i] != labels[j]:  # only compare across classes
                    iou = self.compute_iou(boxes[i], boxes[j])
                    if iou > iou_thresh:
                        if scores[i] >= scores[j]:
                            keep[j] = False
                        else:
                            keep[i] = False

        filtered_boxes = [b for b, k in zip(boxes, keep) if k]
        filtered_scores = [s for s, k in zip(scores, keep) if k]
        filtered_labels = [l for l, k in zip(labels, keep) if k]
        return filtered_boxes, filtered_scores, filtered_labels
    def predict(self, image, return_image=False):
        original_image = image.copy()
        original_w, original_h = original_image.size

        # Resize for prediction
        # Resize just for model input
        yolo_input = image.resize((640, 640))
        ssd_input = image.resize((320, 320))

        # Run models on resized images
        yolo_boxes, yolo_scores, yolo_labels = self.run_yolo(yolo_input, 640, 640)
        ssd_boxes, ssd_scores, ssd_labels = self.run_ssd(ssd_input, 320, 320)

        boxes, scores, labels = weighted_boxes_fusion(
            [yolo_boxes, ssd_boxes],
            [yolo_scores, ssd_scores],
            [yolo_labels, ssd_labels],
            iou_thr=self.iou_thresh,
            skip_box_thr=self.score_thresh
        )
        boxes, scores, labels = self.suppress_conflicting_classes(boxes, scores, labels, iou_thresh=0.7)

        if return_image:
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("arial.ttf", size=20)
            except:
                font = ImageFont.load_default()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = [coord * dim for coord, dim in zip(box, [original_w, original_h, original_w, original_h])]
                draw.rectangle([x1, y1, x2, y2], outline="green", width=5)
                draw.text((x1, y1 - 20), f"{self.names[int(label)]} {score:.2f}", font=font, fill="red")
            return image
        else:
            return boxes, scores, labels
        

