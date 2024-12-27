import cv2
import os
import random
from glob import glob
from cv2 import aruco
import uuid
import shutil

# Parameters
INPUT_DIR = "/home/sam/bucket/Ants/trials/20241108_1/"
OUTPUT_DIR = "/home/sam/bucket/sam/ant_tracking/aruco_imgs/train_dataset"
CROP_SIZE = 128  # Fixed crop size. Needs to be 224 for resnet, watch image aug
FRAMES_PER_VIDEO = 500  # Number of random frames to extract from each video

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
detectParams = aruco.DetectorParameters()
detectParams.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
detectParams.adaptiveThreshConstant = 3
detectParams.adaptiveThreshWinSizeMin = 10
detectParams.adaptiveThreshWinSizeMax = 40
detectParams.adaptiveThreshWinSizeStep = 10
detectParams.errorCorrectionRate = 1
detector = aruco.ArucoDetector(aruco_dict, detectParams)

def remove_small_folders(output_dir, min_images):
    """
    Remove folders with fewer than `min_images` in the output directory.
    Args:
        output_dir: Path to the root directory containing subfolders.
        min_images: Minimum number of images required to keep a folder.
    """
    for subfolder in os.listdir(output_dir):
        subfolder_path = os.path.join(output_dir, subfolder)
        if os.path.isdir(subfolder_path):
            num_images = len([file for file in os.listdir(subfolder_path) if file.endswith(('.png', '.jpg', '.jpeg'))])
            if num_images < min_images:
                shutil.rmtree(subfolder_path)
                print(f"Removed folder: {subfolder_path} (contained {num_images} images)")

def save_fixed_crops(image, corners, ids, output_dir, crop_size):
    """
    Save 128x128 crops of detected ArUco tags into a structured directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, corner in enumerate(corners):
        tag_id = int(ids[i][0])  # Get the ID of the tag
        tag_dir = os.path.join(output_dir, str(tag_id))
        os.makedirs(tag_dir, exist_ok=True)

        # Calculate the center
        x_center = int(corner[0][:, 0].mean())
        y_center = int(corner[0][:, 1].mean())
        
        # Fixed-size crop boundaries
        x_min = max(0, x_center - crop_size // 2)
        x_max = x_min + crop_size
        y_min = max(0, y_center - crop_size // 2)
        y_max = y_min + crop_size
        
        # Ensure the crop is within image bounds
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)
        
        # Crop the image
        fixed_crop = image[y_min:y_max, x_min:x_max]
        fixed_crop = cv2.resize(fixed_crop, (crop_size, crop_size))

        # Save the crop
        random_name = str(uuid.uuid4())  # Generates a unique identifier
        crop_path = os.path.join(tag_dir, f"{random_name}.png")
        cv2.imwrite(crop_path, fixed_crop)
        print(f"Saved: {crop_path}")


def process_video(video_path, output_dir, crop_size, frames_per_video):
    """
    Extract random frames from a video and detect ArUco markers.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Processing video: {os.path.basename(video_path)}, Frames: {total_frames}")

    # Randomly select frames
    selected_frames = random.sample(range(total_frames), min(frames_per_video, total_frames))

    for frame_idx in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx} in {video_path}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            print(f"Detected {len(ids)} ArUco tags in frame {frame_idx} of {os.path.basename(video_path)}.")
            save_fixed_crops(frame, corners, ids, output_dir, crop_size)
        else:
            print(f"No ArUco tags detected in frame {frame_idx} of {os.path.basename(video_path)}.")

    cap.release()


def process_video_directory(input_dir, output_dir, crop_size, frames_per_video):
    """
    Process all videos in the input directory to extract ArUco markers.
    """
    video_paths = glob(os.path.join(input_dir, "*.avi"))  # Adjust file extension if needed
    for video_path in video_paths:
        process_video(video_path, output_dir, crop_size, frames_per_video)


# Run the script
process_video_directory(INPUT_DIR, OUTPUT_DIR, CROP_SIZE, FRAMES_PER_VIDEO)

remove_small_folders(OUTPUT_DIR,50)
