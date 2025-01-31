import cv2
import numpy as np
import os

# Define paths to YOLO files
weights_path = "d:/downloads d/cg mini proj/yolov3.weights"
config_path = "d:/downloads d/cg mini proj/yolov3.cfg"
names_path = "d:/downloads d/cg mini proj/coco.names"

# Check if files exist
if not os.path.exists(weights_path):
    print(f"Error: {weights_path} not found.")
    exit()
if not os.path.exists(config_path):
    print(f"Error: {config_path} not found.")
    exit()
if not os.path.exists(names_path):
    print(f"Error: {names_path} not found.")
    exit()

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the custom class labels
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define a video capture object
video_path = "d:/downloads d/cg mini proj/video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Create a 4D blob from a frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    number_plate_detected = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if confidence is above threshold and it's either helmet or number plate
            if confidence > 0.5 and (class_id == classes.index('helmet') or class_id == classes.index('number_plate')):
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Check if number plate is detected
                if class_id == classes.index('number_plate'):
                    number_plate_detected = True

    if not number_plate_detected:
        cv2.putText(frame, "No number plate detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0) if label == 'helmet' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
