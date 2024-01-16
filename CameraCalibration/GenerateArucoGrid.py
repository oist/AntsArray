import cv2
import numpy as np

aruco_type = cv2.aruco.DICT_5X5_1000

def create_aruco_marker(marker_id, marker_size, aruco_dict_type=aruco_type):
    """
    Create an ArUco marker with a given ID and size.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_image, 1)
    return marker_image

def get_dictionary_name(aruco_dict_type):
    """
    Map the ArUco dictionary type to its string name.
    """
    dictionary_mapping = {
        cv2.aruco.DICT_4X4_50: "DICT_4X4_50",
        cv2.aruco.DICT_4X4_100: "DICT_4X4_100",
        cv2.aruco.DICT_4X4_250: "DICT_4X4_250",
        cv2.aruco.DICT_4X4_1000: "DICT_4X4_1000",
        cv2.aruco.DICT_5X5_50: "DICT_5X5_50",
        cv2.aruco.DICT_5X5_100: "DICT_5X5_100",
        cv2.aruco.DICT_5X5_250: "DICT_5X5_250",
        cv2.aruco.DICT_5X5_1000: "DICT_5X5_1000",
        cv2.aruco.DICT_6X6_50: "DICT_6X6_50",
        cv2.aruco.DICT_6X6_100: "DICT_6X6_100",
        cv2.aruco.DICT_6X6_250: "DICT_6X6_250",
        cv2.aruco.DICT_6X6_1000: "DICT_6X6_1000",
        cv2.aruco.DICT_7X7_50: "DICT_7X7_50",
        cv2.aruco.DICT_7X7_100: "DICT_7X7_100",
        cv2.aruco.DICT_7X7_250: "DICT_7X7_250",
        cv2.aruco.DICT_7X7_1000: "DICT_7X7_1000",
        cv2.aruco.DICT_ARUCO_ORIGINAL: "DICT_ARUCO_ORIGINAL",
        cv2.aruco.DICT_APRILTAG_16h5: "DICT_APRILTAG_16h5",
        cv2.aruco.DICT_APRILTAG_25h9: "DICT_APRILTAG_25h9",
        cv2.aruco.DICT_APRILTAG_36h10: "DICT_APRILTAG_36h10",
        cv2.aruco.DICT_APRILTAG_36h11: "DICT_APRILTAG_36h11"
    }

    return dictionary_mapping.get(aruco_dict_type, "UnknownDict")

# Grid parameters
markers_x = 29 * 4 # Number of markers in the x direction
markers_y = 41 * 4 # Number of markers in the y direction
marker_size = 8  # Size of the marker in pixels
spacing = 1 * marker_size     # Spacing between markers in pixels

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

# Determine the file name based on the ArUco dictionary type
dict_name = get_dictionary_name(aruco_type)
file_name = f'aruco_grid_{dict_name}.png'

# Save to file
cv2.imwrite(file_name, grid_image)

# Display the grid
cv2.imshow('ArUco Grid', grid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
