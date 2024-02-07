import cv2
import cv2.aruco as aruco

def detect_aruco_tags(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize the detector parameters using default values
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    parameters = aruco.DetectorParameters()

    # Detect the markers in the image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Check if there are any ArUco markers detected
    if len(corners) > 0:
        # Draw the detected markers and their IDs
        aruco.drawDetectedMarkers(img, corners, ids)

        # Display the output image
        cv2.imshow('Detected ArUco tags', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No ArUco markers found")

# Replace the path with the actual path to your image file
detect_aruco_tags(r"C:\Users\machi\Desktop\Image__2024-02-02__16-07-24.tiff")
