import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import argparse
import pickle
import os

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # Replace the final fully connected layer to match training setup
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, args.num_tags)
    )

    # Send the model to the device
    model = model.to(device)

    # Load the trained weights
    model.load_state_dict(torch.load(args.model_name, map_location=device))
    model.eval()

    # Preprocessing transform for the input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load SLEAP detections
    df = pd.read_csv(args.sleap_file)
    df['Frame'] = df['Frame'] - 1  # Convert to 0-based indexing
    df = df.drop(['Score_node'], axis=1)

    # Open video file
    cap = cv2.VideoCapture(args.video_file)
    if not cap.isOpened():
        print(f"Failed to open video file: {args.video_file}")
        exit()

    #get cam num
    parts = os.path.basename(args.video_file).split('cam')
    if len(parts) > 2:
        cam_number = parts[2][:2]  # Take the first two characters after the second 'cam'
    else:
        raise ValueError("Filename does not have the required format with two 'cam' occurrences.")
        
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    rows = []  # To store rows for DataFrame
    
    # Process video frame by frame
    for frame_idx in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx}.")
            break
    
        # Adjust brightness for visualization
        if args.visualize:
            disp_img = np.clip(frame.astype(np.float32) * args.brightness_factor, 0, 255).astype(np.uint8)
    
        # Get SLEAP detections for the current frame
        sleap_detections = df[df['Frame'] == frame_idx][['X', 'Y']].to_numpy()
    
        valid_crops = []
        crop_positions = []
    
        # Collect valid crops for the current frame
        for x, y in sleap_detections:
            if np.isnan(x) or np.isnan(y):
                continue
    
            x_start = max(int(x - args.crop_size // 2), 0)
            y_start = max(int(y - args.crop_size // 2), 0)
            x_end = min(int(x + args.crop_size // 2), frame_width)
            y_end = min(int(y + args.crop_size // 2), frame_height)
    
            cropped_region = frame[y_start:y_end, x_start:x_end]
            if cropped_region.shape[0] != args.crop_size or cropped_region.shape[1] != args.crop_size:
                cropped_region = cv2.resize(cropped_region, (args.crop_size, args.crop_size))
    
            cropped_region_pil = Image.fromarray(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
            cnn_input = transform(cropped_region_pil)
            valid_crops.append(cnn_input)
            crop_positions.append((x, y))
    
        # Process crops in mini-batches
        for start_idx in range(0, len(valid_crops), args.batch_size):
            end_idx = start_idx + args.batch_size
            batch_crops = valid_crops[start_idx:end_idx]
            batch_positions = crop_positions[start_idx:end_idx]
    
            cnn_inputs = torch.stack(batch_crops).to(device)
            with torch.no_grad():
                outputs = model(cnn_inputs)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_ids = torch.max(probabilities, dim=1)
    
                confidence = confidence.cpu().numpy()
                predicted_ids = predicted_ids.cpu().numpy()
    
                # Apply confidence threshold
                predicted_ids = np.where(confidence < args.confidence_threshold, -1, predicted_ids)
    
                # Store detections in the dictionary and prepare rows for DataFrame
                for (x, y), predicted_id in zip(batch_positions, predicted_ids):
                    if predicted_id != -1:  # Only store valid IDs

                        # Append a row for DataFrame
                        rows.append({
                            'X': x,
                            'Y': y,
                            'Frame': frame_idx,
                            'ARUCO_number': predicted_id,
                            'Cam': int(cam_number)-1
                        })

        if args.visualize:
            for (x, y), predicted_id in zip(crop_positions, predicted_ids):
                cv2.circle(disp_img, (int(x), int(y)), 20, (0, 0, 255), -1)
                cv2.putText(disp_img, f"ID: {predicted_id}", (int(x) + 15, int(y) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    
            cv2.putText(disp_img, f"Frame {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
            show_img = cv2.resize(disp_img, (1080, 720))
            cv2.imshow("ArUco Tag Detection", show_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            
        # Convert rows to a DataFrame
    df_aruco = pd.DataFrame(rows)
    df_aruco.to_pickle(args.output_file)
    print('wrote file to ' + args.output_file)


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames with a ResNet-based model.")
    parser.add_argument("--video_file", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--sleap_file", type=str, required=True, help="Path to the SLEAP CSV file.")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output pkl file.")
    parser.add_argument("--crop_size", type=int, default=128, help="Size of the crop around the detection.")
    parser.add_argument("--brightness_factor", type=float, default=1.2, help="Brightness adjustment factor for visualization.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing crops.")
    parser.add_argument("--num_tags", type=int, default=111, help="Number of tags (classes) for the model.")
    parser.add_argument("--confidence_threshold", type=float, default=0.99, help="Confidence threshold for predictions.")
    parser.add_argument("--visualize", type=int, default=0, help="Enable visualization (1 for yes, 0 for no).")

    args = parser.parse_args()
    main(args)
