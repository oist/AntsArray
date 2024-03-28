import cv2
import cv2.aruco as aruco


def detect_aruco_tags(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

    # Create default parameters for detection
    parameters = aruco.DetectorParameters()

    # Adjust detection parameters
    # Default is 7, adjust based on lighting conditions
    parameters.adaptiveThreshConstant = 3
    parameters.adaptiveThreshWinSizeMin = 10
    parameters.adaptiveThreshWinSizeMax = 40
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.errorCorrectionRate = 1

    # Create an instance of ArucoDetector with custom parameters
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # Detect the markers in the image
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    # Check if there are any ArUco markers detected
    if ids is not None and len(corners) > 0:
        # Draw the detected markers and their IDs
        aruco.drawDetectedMarkers(img, corners, ids)

    # Resize for display if necessary
    resized_img = cv2.resize(
        img, (1600, int(1600 * (img.shape[0] / img.shape[1]))))

    # Display the output image
    cv2.imshow('Detected ArUco tags', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Replace the path with the actual path to your image file
detect_aruco_tags(
    r"Z:\ReiterU\Ants\basler\Anouk\Image__2024-03-22__17-32-08.tiff")
