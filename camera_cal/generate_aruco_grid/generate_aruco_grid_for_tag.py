import cv2
import cv2.aruco as aruco
import numpy as np
import argparse

def generate_aruco_grid(dictionary_name, rows, cols, tag_size=100, margin=10, output_file='aruco_grid.png'):
    # Load the dictionary
    aruco_dict = aruco.getPredefinedDictionary(dictionary_name)

    # Calculate the size of the output image
    grid_width = cols * tag_size + (cols + 1) * margin
    grid_height = rows * tag_size + (rows + 1) * margin

    # Create a blank white image
    grid_image = 255 * np.ones((grid_height, grid_width), dtype=np.uint8)

    # Generate and draw the ArUco tags on the grid
    for r in range(rows):
        for c in range(cols):
            tag_id = r * cols + c
            # Generate the tag image using drawMarker (available in OpenCV 4.5+; generateImageMarker requires 4.7+)
            tag_image = aruco.drawMarker(aruco_dict, tag_id, tag_size)
            
            # Define where the tag should be placed on the grid
            start_x = c * tag_size + (c + 1) * margin
            start_y = r * tag_size + (r + 1) * margin
            
            # Place the tag on the grid image
            grid_image[start_y:start_y + tag_size, start_x:start_x + tag_size] = tag_image

    # Save the output image
    cv2.imwrite(output_file, grid_image)
    print(f"Grid of ArUco tags saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a grid of ArUco tags')
    parser.add_argument('--rows', type=int, required=True, help='Number of rows in the grid')
    parser.add_argument('--cols', type=int, required=True, help='Number of columns in the grid')
    parser.add_argument('--dict', type=str, required=True, help='ArUco dictionary name (e.g., DICT_4X4_250)')
    parser.add_argument('--tag_size', type=int, default=100, help='Size of each tag in pixels (default: 100)')
    parser.add_argument('--margin', type=int, default=10, help='Margin between tags in pixels (default: 10)')
    parser.add_argument('--output_file', type=str, default='aruco_grid.png', help='Name of the output file (default: aruco_grid.png)')
    args = parser.parse_args()

    # Convert dictionary name to ArUco dictionary type
    try:
        aruco_dict = getattr(aruco, args.dict)
    except AttributeError:
        print(f"Invalid dictionary name: {args.dict}")
        exit(1)

    # Generate the grid
    generate_aruco_grid(aruco_dict, args.rows, args.cols, args.tag_size, args.margin, args.output_file)
