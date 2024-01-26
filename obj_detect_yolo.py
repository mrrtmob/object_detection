import cv2
from yolov5 import YOLOv5
import yaml

# Load the YOLOv5 model
model = YOLOv5("yolov5s.pt", device="cpu")  # Use "cuda" for GPU or "cpu"

# Load class names
with open("coco.yaml", 'r') as f:
    class_names = yaml.safe_load(f)['names']

# Start video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection (xyxy format)
    results = model.predict(frame)

    # Draw bounding boxes and labels on the frame
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        label = class_names[int(cls)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
