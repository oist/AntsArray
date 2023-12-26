import cv2
import numpy as np

def create_aruco_marker(marker_id, marker_size, aruco_dict_type=cv2.aruco.DICT_5X5_1000):
    """
    Create an ArUco marker with a given ID and size.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_image, 1)
    return marker_image

# Grid parameters
markers_x = 29  # Number of markers in the x direction
markers_y = 41  # Number of markers in the y direction
marker_size = 7  # Size of the marker in pixels
spacing = 4 * marker_size     # Spacing between markers in pixels

# Create a blank image for the grid
grid_width = markers_x * (marker_size + spacing)
grid_height = markers_y * (marker_size + spacing)
grid_image = np.ones((grid_height, grid_width), dtype=np.uint8) * 255  # white background

# Fill the grid with markers
marker_id = 0
for y in range(markers_y):
    for x in range(markers_x):
        # Ensure marker_id is within the range of the dictionary
        if marker_id >= 1000:
            marker_id = 0
        marker = create_aruco_marker(marker_id, marker_size)
        start_x = x * (marker_size + spacing)
        start_y = y * (marker_size + spacing)
        grid_image[start_y:start_y+marker_size, start_x:start_x+marker_size] = marker
        marker_id += 1  # Increment marker ID

# Display the grid
cv2.imshow('ArUco Grid', grid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save to file
cv2.imwrite('aruco_grid.png', grid_image)
