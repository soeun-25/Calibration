import subprocess
import cv2
import numpy as np

# Define the YOLO command
yolo_command = r"yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='final_calibration.jpg' save=True"

# Run the YOLO command using subprocess and capture the output
result = subprocess.run(yolo_command, shell=True, capture_output=True)

# Get the output from YOLO
output_lines = result.stdout.decode().split('\n')

# Initialize a list to store bounding box coordinates
bounding_boxes = []

# Iterate through the output lines
for line in output_lines:
    if line.startswith('person'):  # Assuming 'person' is the class label
        _, confidence, x, y, w, h = map(float, line.split()[1:])
        
        # Extracting the coordinates of the top-left corner
        top_left_x = int(x)
        top_left_y = int(y)

        bounding_boxes.append((top_left_x, top_left_y, int(w), int(h)))

# Now 'bounding_boxes' contains the coordinates of the top-left corners of the detected persons' bounding boxes
# You can use this list as needed in your code
print("Bounding Box Coordinates:", bounding_boxes)

# import subprocess

# # Define the YOLO command
# yolo_command = "yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='/Users/juhno1023/Downloads/TESTAL.jpg' save=True"

# # Run the YOLO command using subprocess and capture the output
# result = subprocess.run(yolo_command, shell=True, capture_output=True)

# # Print the output
# print(result.stdout.decode())


# from ultralytics import YOLO
# import numpy

# model = YOLO("yolov8n.pt", "v8")

# detection_output = model.predict(source ="inference/images/img0.JPG", conf=0.25,save=False)

# print(detection_output)

# print(detection_output[0].numpy())