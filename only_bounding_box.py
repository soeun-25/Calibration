import os
import math
import cv2

def setLabel(img, pts, label):
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.putText(img, label, (pt1[0], pt1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

image_path = 'runs\detect\predict13\\11_Dark.jpg'

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thermal_path = 'runs\detect\predict14\\thermal_image.jpg'
thermal = cv2.imread(thermal_path)

# Convert the thermal image to grayscale
thermal_frame_v = cv2.cvtColor(thermal, cv2.COLOR_BGR2HSV)[:, :, 2]
# Apply bilateral filter for smoothing
blurred_brightness = cv2.bilateralFilter(thermal_frame_v, 9, 150, 150)
thresh = 50
thermal_edges = cv2.Canny(blurred_brightness, thresh, thresh * 2, L2gradient=True)
contours, _ = cv2.findContours(thermal_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for i, cont in enumerate(contours):
    approx = cv2.approxPolyDP(cont, cv2.arcLength(cont, True) * 0.02, True)
    vtc = len(approx)

    if vtc == 3:
        setLabel(img, cont, 'Tri')
    elif vtc == 4:
        setLabel(img, cont, 'Rec')
        rect = cv2.boundingRect(cont)
                # Print the coordinates of the bounding box
        print(f"Bounding Box {i + 1} Coordinates: Top-Left ({rect[0]}, {rect[1]}), Bottom-Right ({rect[0] + rect[2]}, {rect[1] + rect[3]})")

        # Extract the region of interest (ROI) based on the bounding box
        x, y, w, h = rect
        
        roi = thermal[y:y+h, x:x+w]

        # Save the cropped image
        cv2.imwrite(f'cropped_image_{i + 1}.jpg', roi)
    elif vtc == 5:
        setLabel(img, cont, 'Pen')
    else:
        rect = cv2.boundingRect(cont)
        setLabel(img, cont, 'Cir')


cv2.imshow('img', roi)
cv2.waitKey()
cv2.destroyAllWindows()

