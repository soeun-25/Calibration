# # #https://www.flirkorea.com/developer/lepton-integration/Project-People-finding-with-a-Lepton/

#-------------------------------------------------------------- 경계선 따기
# import cv2
# import numpy as np

# frame_v = cv2.imread(r'C:\\Users\\user\Desktop\\YOLOv8\\thermal_image.jpg')

# blurredBrightness = cv2.bilateralFilter(frame_v,9,150,150)
# thresh = 70
# edges = cv2.Canny(blurredBrightness,thresh,thresh*2, L2gradient=True)

# cv2.imshow("preview", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# --------------------------------------- 경계선을 저조도 이미지에 접목
# import cv2
# import numpy as np

# # Load the thermal image
# thermal_image_path = 'C:\\Users\\user\Desktop\\YOLOv8\\thermal_image.jpg'
# thermal_frame = cv2.imread(thermal_image_path)

# # Convert the thermal image to grayscale
# thermal_frame_v = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2HSV)[:, :, 2]

# # Apply bilateral filter for smoothing
# blurred_brightness = cv2.bilateralFilter(thermal_frame_v, 9, 150, 150)

# # Set threshold for Canny Edge Detection
# thresh = 50
# thermal_edges = cv2.Canny(blurred_brightness, thresh, thresh * 2, L2gradient=True)

# # Create a binary mask based on brightness threshold
# _, thermal_mask = cv2.threshold(blurred_brightness, 200, 1, cv2.THRESH_BINARY)
# erode_size = 5
# dilate_size = 7
# eroded = cv2.erode(thermal_mask, np.ones((erode_size, erode_size)))
# thermal_mask = cv2.dilate(eroded, np.ones((dilate_size, dilate_size)))

# # Load the second image
# image_path = 'C:\\Users\\user\Desktop\\YOLOv8\\1228_15_45_1076.jpg'
# frame = cv2.imread(image_path)

# # Resize the thermal edges to match the size of the second image
# thermal_edges_resized = cv2.resize(thermal_edges, (frame.shape[1], frame.shape[0]))

# # Combine the thermal edges with the second image
# result_image = cv2.cvtColor(thermal_edges_resized * thermal_mask, cv2.COLOR_GRAY2RGB) | frame

# # Display the result
# cv2.imshow("preview", cv2.resize(result_image, (640, 480), interpolation=cv2.INTER_CUBIC))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#---------------------------------------- 최종 성능이 좋은 합성 코드
# import cv2
# import numpy as np

# # Load the thermal image
# thermal_image_path = 'C:\\Users\\user\Desktop\\YOLOv8\\thermal_image.jpg'
# thermal_frame = cv2.imread(thermal_image_path)

# # Convert the thermal image to grayscale
# thermal_frame_v = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2HSV)[:, :, 2]

# # Apply bilateral filter for smoothing
# blurred_brightness = cv2.bilateralFilter(thermal_frame_v, 9, 150, 150)

# # Set threshold for Canny Edge Detection
# thresh = 50
# thermal_edges = cv2.Canny(blurred_brightness, thresh, thresh * 2, L2gradient=True)

# # Create a binary mask based on brightness threshold
# _, thermal_mask = cv2.threshold(blurred_brightness, 200, 255, cv2.THRESH_BINARY)
# erode_size = 5
# dilate_size = 7
# eroded = cv2.erode(thermal_mask, np.ones((erode_size, erode_size)))
# thermal_mask = cv2.dilate(eroded, np.ones((dilate_size, dilate_size)))

# # Find contours on the thermal mask
# contours, _ = cv2.findContours(thermal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Create an image filled with black color
# filled_image = np.zeros_like(thermal_frame)

# # Fill the contours with white color
# cv2.fillPoly(filled_image, contours, (255, 255, 255))

# # Load the second image
# image_path = 'C:\\Users\\user\Desktop\\YOLOv8\\1228_15_45_1076.jpg'
# frame = cv2.imread(image_path)

# # Resize the thermal edges to match the size of the second image
# thermal_edges_resized = cv2.resize(thermal_edges, (frame.shape[1], frame.shape[0]))

# # Convert the grayscale image to 3-channel
# thermal_edges_resized_rgb = cv2.cvtColor(thermal_edges_resized, cv2.COLOR_GRAY2RGB)

# # Combine the thermal edges with the second image
# result_image = cv2.bitwise_and(frame, cv2.bitwise_not(filled_image)) + thermal_edges_resized_rgb

# # Display the result
# cv2.imshow("preview", cv2.resize(result_image, (640, 480), interpolation=cv2.INTER_CUBIC))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#------ 깔끔한 경계선 합성 코드 
#참조 = https://www.flirkorea.com/developer/lepton-integration/Project-People-finding-with-a-Lepton/

import cv2
import numpy as np

# Load the thermal image
thermal_image_path = 'thermal_image.jpg'
thermal_frame = cv2.imread(thermal_image_path)

# Convert the thermal image to grayscale
thermal_frame_v = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2HSV)[:, :, 2]

# Apply bilateral filter for smoothing
blurred_brightness = cv2.bilateralFilter(thermal_frame_v, 9, 150, 150)

# Set threshold for Canny Edge Detection
thresh = 20
thermal_edges = cv2.Canny(blurred_brightness, thresh, thresh * 2, L2gradient=True)

# Create a binary mask based on brightness threshold
_, thermal_mask = cv2.threshold(blurred_brightness, 200, 255, cv2.THRESH_BINARY)
erode_size = 20
dilate_size = 7
eroded = cv2.erode(thermal_mask, np.ones((erode_size, erode_size)))
thermal_mask = cv2.dilate(eroded, np.ones((dilate_size, dilate_size)))

# Find contours only on the original thermal edges
contours_edges, _ = cv2.findContours(thermal_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an image filled with black color for contours only
contours_only_image = np.zeros_like(thermal_frame)

# Draw the contours on the contours-only image
cv2.drawContours(contours_only_image, contours_edges, -1, (255, 255, 255), 2)

# Load the second image
low_light_image_path = '2_Dark.jpg'
low_light_frame = cv2.imread(low_light_image_path)

high_light_image_path = '2.jpg'
high_light_frame = cv2.imread(high_light_image_path) 

# Overlay contours on the original image
result_low_light_image = cv2.bitwise_and(low_light_frame, cv2.bitwise_not(contours_only_image)) + contours_only_image
result_high_light_image = cv2.bitwise_and(high_light_frame, cv2.bitwise_not(contours_only_image)) + contours_only_image

cv2.imwrite('final_calibration_thresh'+ str(thresh) +'.jpg',result_low_light_image)
cv2.imwrite('contours_only_image.jpg',contours_only_image)

# Display the result
cv2.imshow("Thermal image", cv2.resize(thermal_frame, (740, 580)))
cv2.imshow("Original low-Light image", cv2.resize(low_light_frame, (740, 580)))
cv2.imshow("Original Image with Thermal Edges", cv2.resize(contours_only_image, (740, 580), interpolation=cv2.INTER_CUBIC))
cv2.imshow("Final Low Light Image", cv2.resize(result_low_light_image, (740, 580), interpolation=cv2.INTER_CUBIC))
cv2.imshow("Final High Light Image", cv2.resize(result_high_light_image, (740, 580), interpolation=cv2.INTER_CUBIC))
cv2.waitKey(0)
cv2.destroyAllWindows()

#------------------------------ 경계선 및 결합 이미지 생성 코드 
# import cv2
# import numpy as np

# # Load the thermal image
# thermal_image_path = 'C:\\Users\\user\Desktop\\YOLOv8\\thermal_image.jpg'
# thermal_frame = cv2.imread(thermal_image_path)

# # Convert the thermal image to grayscale
# thermal_frame_v = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2HSV)[:, :, 2]

# # Apply bilateral filter for smoothing
# blurred_brightness = cv2.bilateralFilter(thermal_frame_v, 9, 150, 150)

# # Set threshold for Canny Edge Detection
# thresh = 50
# thermal_edges = cv2.Canny(blurred_brightness, thresh, thresh * 2, L2gradient=True)

# # Create a binary mask based on brightness threshold
# _, thermal_mask = cv2.threshold(blurred_brightness, 200, 255, cv2.THRESH_BINARY)
# erode_size = 5
# dilate_size = 7
# eroded = cv2.erode(thermal_mask, np.ones((erode_size, erode_size)))
# thermal_mask = cv2.dilate(eroded, np.ones((dilate_size, dilate_size)))

# # Find contours on the thermal mask
# contours, _ = cv2.findContours(thermal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Create an image filled with black color
# filled_image = np.zeros_like(thermal_frame)

# # Fill the contours with white color
# cv2.fillPoly(filled_image, contours, (255, 255, 255))

# # Load the second image
# image_path = 'C:\\Users\\user\Desktop\\YOLOv8\\1228_15_45_1076.jpg'
# frame = cv2.imread(image_path)

# # Resize the thermal edges to match the size of the second image
# thermal_edges_resized = cv2.resize(thermal_edges, (frame.shape[1], frame.shape[0]))

# # Convert the grayscale image to 3-channel
# thermal_edges_resized_rgb = cv2.cvtColor(thermal_edges_resized, cv2.COLOR_GRAY2RGB)

# # Combine the thermal edges with the second image
# result_image_with_edges = cv2.bitwise_and(frame, cv2.bitwise_not(filled_image)) + thermal_edges_resized_rgb

# # Display the results
# cv2.imshow("Original Image with Thermal Edges", cv2.resize(result_image_with_edges, (640, 480), interpolation=cv2.INTER_CUBIC))
# cv2.imshow("Thermal Edges Only", cv2.resize(thermal_edges_resized_rgb, (640, 480), interpolation=cv2.INTER_CUBIC))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
