import cv2
import numpy as np

high_light_image_path = 'C:\\Users\\user\Desktop\\YOLOv8\\1228_15_45_1083.jpg'
high_light_frame = cv2.imread(high_light_image_path) 

blurredBrightness = cv2.bilateralFilter(high_light_frame,9,150,150)
thresh = 15
edges = cv2.Canny(blurredBrightness,thresh,thresh*2, L2gradient=True)

# Convert edges to a 3-channel image with red color
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
edges_colored[:, :, 1] = 0  # Set green channel to 0
edges_colored[:, :, 2] = 0  # Set blue channel to 255 (red color)

cv2.imwrite('high_light_edges.jpg',edges_colored)

high_light_edges_path = 'C:\\Users\\user\Desktop\\YOLOv8\\high_light_edges.jpg'
high_light_edges = cv2.imread(high_light_edges_path)

low_light_edges_path = 'C:\\Users\\user\Desktop\\YOLOv8\\contours_only_image.jpg'
low_light_edges = cv2.imread(low_light_edges_path) 

final = high_light_edges + low_light_edges
cv2.imwrite('final.jpg',final)
cv2.imshow("high_light_edges", cv2.resize(final, (740, 580)))
cv2.waitKey(0)
cv2.destroyAllWindows()


