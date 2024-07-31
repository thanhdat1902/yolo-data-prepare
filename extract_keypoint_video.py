import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from collections import deque
import torch
# (14,3), (16,3), (17,3), (19,1), (20,3), (22,3), (23,3), (24,3), 
arr = [(26,3), (28,2)]
frame_id_start = 180366
model = YOLO("yolov9e.pt")
model1 = YOLO("./best.pt")

threshold_shoulder_movement = 60
def get_current_set(na):
    if 13 < na <= 23:
        return "train"
    elif 23 < na < 25:
        return "valid"
    else:
        return "test"

shoulders_buffer = deque(maxlen=5)  # Buffer size can be adjusted
chest_buffer = deque(maxlen=5)

for na, ord in arr:
    current_set = get_current_set(na)
    for i in range(ord):
        cap = cv2.VideoCapture('./video/Background/p{}_front_{}_white.mp4'.format(na, i+1))
        cap1 = cv2.VideoCapture('./video/Need Convert/p{}_front_{}.mp4'.format(na, i+1))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        name = "P{}-{}".format(na, i+1)
        previous_shoulders = None
        print("{}-{}".format(na,ord))


        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            
            while True:
                ret, image = cap.read()
                ret1, image1 = cap1.read()
                pbar.update(1)
                # image = cv2.imread("p14_front_1 - Trim_white.jpg")
                if not (ret and ret1):
                    break
                
                height, width, _ = image.shape
                results = model.predict(image, classes=[0,1],  verbose=False)

                
                # Get the bounding boxes for detected persons (class id 0 typically represents 'person')
                bboxes = []
                for result in results[0].boxes:
                    if result.cls == 0:  # Assuming class '0' is 'person'
                        x_min, y_min, x_max, y_max = result.xyxy[0]
                        x_min = int(x_min)
                        y_min = int(y_min)
                        x_max = int(x_max)
                        y_max = int(y_max)
                        bboxes.append((x_min, y_min, x_max, y_max))

                # Find the largest bounding box near the center of the image
                center_x, center_y = width / 2, height / 2
                center_bbox = None
                max_area = 0

                for bbox in bboxes:
                    x_min, y_min, x_max, y_max = bbox
                    bbox_center_x = (x_min + x_max) / 2
                    bbox_center_y = (y_min + y_max) / 2
                    distance = np.sqrt((center_x - bbox_center_x) ** 2 + (center_y - bbox_center_y) ** 2)
                    area = (x_max - x_min) * (y_max - y_min)

                    if distance < min(width, height) / 2 and area > max_area:  # Adjust distance threshold if needed
                        max_area = area
                        center_bbox = bbox

                if center_bbox:
                    x_min, y_min, x_max, y_max = center_bbox
                    # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box

                    # Crop the image to the center person bounding box
                    cropped_image = image[y_min:y_max, x_min:x_max]

                    # Perform keypoint detection on the cropped image
                    keypoints_results = model1.predict(cropped_image,  verbose=False)


                    # Extract keypoints from the detection results
                    keypoints = []
                    for result in keypoints_results[0].boxes:
                        x_min_kp, y_min_kp, x_max_kp, y_max_kp = result.xyxy[0]
                        keypoints.append(((x_min_kp + x_max_kp) / 2, (y_min_kp + y_max_kp) / 2))  # Center of bbox as keypoint

                    # Filter and identify the keypoints
                    if len(keypoints) >= 4:
                        # Sort by y-axis (ascending)
                        keypoints.sort(key=lambda kp: kp[1])

                        # Identify shoulders (two highest points)
                        shoulders = keypoints[:2]
                        shoulders.sort(key=lambda kp: kp[0])  # Sort shoulders by x-axis

                        # The third keypoint is assumed to be the chest
                        chest = keypoints[2]

                        if shoulders[0][0] < chest[0] < shoulders[1][0]:
                            if previous_shoulders is not None:
                                # Compute distances between current and previous shoulders
                                left_shoulder_dist = torch.sqrt(torch.pow(shoulders[0][0] -  previous_shoulders[0][0], 2) + torch.pow(shoulders[0][1] -  previous_shoulders[0][1], 2))
                                right_shoulder_dist = torch.sqrt(torch.pow(shoulders[1][0] -  previous_shoulders[1][0], 2) + torch.pow(shoulders[1][1] -  previous_shoulders[1][1], 2))

                                if left_shoulder_dist > threshold_shoulder_movement or right_shoulder_dist > threshold_shoulder_movement:
                                    continue  # Skip this frame if shoulders are too far apart from previous frame

                            # Update previous shoulders
                            previous_shoulders = shoulders

                            # Identify potential belly button from remaining keypoints
                            remaining_keypoints = keypoints[3:]
                            belly_button = None        
                            distance_shoulders = abs(shoulders[1][0] - shoulders[0][0])
                            threshold_middle = 20
                            for kp in remaining_keypoints:                
                                # Calculate distances
                                distance_belly_chest = abs(kp[1] - chest[1])  # Vertical distance between belly button and chest

                                # Proportional Filtering
                                proportion_threshold = 0.7  # Example value
                                if distance_belly_chest > (distance_shoulders* proportion_threshold):
                                    if abs(kp[0] - chest[0]) < threshold_middle and shoulders[0][0] < kp[0] < shoulders[1][0]:
                                        belly_button = kp
                                        break
                            

                            if belly_button is not None:
                                yolo_keypoints = [( (kp[0] + x_min) / width, (kp[1] + y_min) / height) for kp in [shoulders[0], shoulders[1], chest, belly_button]]
                                keypoints_str = " ".join("{:07f} {:07f} 2".format(kp[0], kp[1]) for kp in yolo_keypoints)
                                
                                class_index = 0  # Assuming 'person' class index is 0
                                x_center = (x_min + x_max) / 2 / width
                                y_center = (y_min + y_max) / 2 / height
                                bbox_width = (x_max - x_min) / width
                                bbox_height = (y_max - y_min) / height
                                yolo_format = f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height} {keypoints_str}"

                                # Write YOLO format to a text file
                                txt_file_path =  "./datasets_new/{}/labels/frame_{:07d}.txt".format(current_set, frame_id_start)
                                with open(txt_file_path, 'w') as f:
                                    f.write(yolo_format)


                                # Save the image with bounding box and keypoints
                                output_path = "./datasets_new/{}/images/frame_{:07d}.PNG".format(current_set, frame_id_start)
                                cv2.imwrite(output_path, image1)
                                frame_id_start+=1
                                # start+=1

                                # Draw keypoints on the original image
                                # for kp in yolo_keypoints:
                                #     cv2.circle(image, (int(kp[0] * width), int(kp[1] * height)), 5, (255, 0, 0), -1)  # Blue dots
                            else:
                                # print("Skipping image due to incorrect keypoints alignment")
                                continue
                        else:
                            # print("Skipping image due to incorrect keypoints alignment")
                            continue
                    else:
                        # print("Skipping image due to insufficient number of keypoints detected")
                        continue
                



