import cv2
import numpy as np

# Load the thermal image
thermal_image_path = 'thermal_image.jpg'
thermal_frame = cv2.imread(thermal_image_path)  
top_left_x, top_left_y = 0,0
# Convert the thermal image to grayscale
thermal_frame_v = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2HSV)[:, :, 2]

# Apply bilateral filter for smoothing
blurred_brightness = cv2.bilateralFilter(thermal_frame_v, 9, 150, 150)

# Set threshold for Canny Edge Detection
thresh = 20
thermal_edges = cv2.Canny(blurred_brightness, thresh, thresh * 2, L2gradient=True)

# Create a binary mask based on brightness threshold
_, thermal_mask = cv2.threshold(blurred_brightness, 200, 255, cv2.THRESH_BINARY)
erode_size = 10
dilate_size = 7
eroded = cv2.erode(thermal_mask, np.ones((erode_size, erode_size)))
thermal_mask = cv2.dilate(eroded, np.ones((dilate_size, dilate_size)))

# Find contours only on the original thermal edges
contours_edges, _ = cv2.findContours(thermal_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an image filled with black color for contours only
contours_only_image = np.zeros_like(thermal_frame)

# Draw the contours on the contours-only image
cv2.drawContours(contours_only_image, contours_edges, -1, (255, 255, 255), 2)

# Define the amount of shift in x and y coordinates
x_shift = 16 # 양수 값 -> 오른쪽으로 이동
y_shift = 1 # 음수 값 -> 위로 이동

# Apply the translation to each point in all contours
for contour in contours_edges:
    for point in contour:
        point[0][0] += (top_left_x + x_shift)  # Adjust x coordinate
        point[0][1] += (top_left_y + y_shift)  # Adjust y coordinate

# Draw the adjusted contours on the image
high_light_image_path = '11_Dark.jpg' # 저조도 이미지
high_light_frame = cv2.imread(high_light_image_path) 
image_with_contours = high_light_frame.copy()
cv2.drawContours(image_with_contours, contours_edges, -1, (255, 255, 255), 2)  # Adjusted contours in red

# Display the images
cv2.imwrite('final_calibration.jpg',image_with_contours)
cv2.imshow("Adjusted Contours", image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()


