import cv2
import numpy as np
from ultralytics import YOLO

class YOLOSegmentation:
    def __init__(self, model_path):
        # Load YOLO model
        self.model = YOLO(model_path)

    def detect(self, img):
        # Detect objects in the image with segmentation
        results = self.model.predict(source=img, conf=0.4)
        bboxes, class_ids, segmentations, scores = [], [], [], []
        
        for result in results:
            if result.boxes is not None:
                bboxes.append(result.boxes.xyxy.numpy())  # Bounding boxes
                class_ids.append(result.boxes.cls.numpy())  # Class IDs
                segmentations.append(result.masks.data.cpu().numpy())  # Segmentation masks
                scores.append(result.boxes.conf.numpy())  # Confidence scores
        
        return bboxes, class_ids, segmentations, scores

def main():
    # Load YOLO model (YOLOv8L segmentation model)
    model_path = 'yolov8l-seg.pt'
    model = YOLOSegmentation(model_path)
    
    # Open the webcam feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # COCO class labels
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", 
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
              "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", 
              "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
              "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", 
              "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
              "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "TV monitor", 
              "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
              "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, (640, 480))

        # Detect objects and segment
        bboxes, class_ids, segmentations, scores = model.detect(frame)

        # Draw bounding boxes and segmentation masks
        for bbox, class_id, segmentation, score in zip(bboxes, class_ids, segmentations, scores):
            # Draw bounding box and label
            for box, cls_id, conf in zip(bbox, class_id, score):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{labels[int(cls_id)]:s}: {conf.item():.2f}"  # Use .item() to get a scalar
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw segmentation mask
            if segmentation is not None:
                for seg in segmentation:
                    mask = np.zeros_like(frame)
                    mask[seg == 1] = [0, 0, 255]  # Red mask
                    frame = cv2.addWeighted(frame, 1, mask, 0.5, 0)

        # Display the resulting frame
        cv2.imshow('YOLO Segmentation', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
