import cv2
import cv2.aruco as aruco

def detect_aruco_tags(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    
    # Create default parameters for detection
    parameters = aruco.DetectorParameters()

    # Adjust detection parameters
    parameters.adaptiveThreshConstant = 4  # Default is 7, adjust based on lighting conditions
    parameters.minMarkerPerimeterRate = 0.01  # Default is 0.03, adjust for marker size relative to image
    parameters.maxMarkerPerimeterRate = 4.0  # Default is 4.0, adjust to avoid detecting very large markers
    parameters.polygonalApproxAccuracyRate = 0.07  # Default is 0.05, lower values can make detection more precise
    
    # Detect the markers in the image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Check if there are any ArUco markers detected
    if len(corners) > 0:
        # Draw the detected markers and their IDs
        aruco.drawDetectedMarkers(img, corners, ids)

    # Resize for display if necessary
    resized_img = cv2.resize(img, (1600, int(1600 * (img.shape[0] / img.shape[1]))))
    
    # Display the output image
    cv2.imshow('Detected ArUco tags', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace the path with the actual path to your image file
detect_aruco_tags(r"C:\Users\machi\Desktop\Image__2024-02-02__16-07-24.tiff")
