import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO models for detection and segmentation
model = YOLO("yolov9e.pt")
model1 = YOLO("./best.pt")
segmentation_model = YOLO("yolov8n-seg.pt")  # Assuming YOLOv8n-seg is the segmentation model

# Read the image
image = cv2.imread("test_img/frame_210738.PNG")
height, width, _ = image.shape

# Perform person detection
results = model.predict(image, classes=[0, 1])

# Get bounding boxes for detected persons
bboxes = []
for result in results[0].boxes:
    if result.cls == 0:  # Assuming class '0' is 'person'
        x_min, y_min, x_max, y_max = result.xyxy[0]
        bboxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))

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
    if distance < min(width, height) / 2 and area > max_area:
        max_area = area
        center_bbox = bbox

if center_bbox:
    x_min, y_min, x_max, y_max = center_bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Crop the image to the center person bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Segment the person to create a mask
    seg_results = segmentation_model.predict(cropped_image, classes=[0])  # Assuming class '0' is 'person'

    if seg_results[0].masks is not None and len(seg_results[0].masks) > 0:
        mask_data = seg_results[0].masks.data  # Raw mask data
        mask = mask_data[0].cpu().numpy()  # Convert to NumPy array

        # Ensure mask is a binary mask
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)  # Threshold and convert to uint8

        # Resize the mask to the original bounding box size
        resized_mask = cv2.resize(mask, (x_max - x_min, y_max - y_min))

        # Overlay the mask on the original image
        mask_colored = np.zeros_like(image[y_min:y_max, x_min:x_max])
        mask_colored[resized_mask > 0] = [0, 0, 255]  # Red color for the mask
        blended_image = cv2.addWeighted(cropped_image, 0.7, mask_colored, 0.3, 0)
        image[y_min:y_max, x_min:x_max] = blended_image

        # Perform keypoint detection on the cropped image
        keypoints_results = model1.predict(cropped_image)

        # Extract keypoints and filter based on mask
        keypoints = []
        for result in keypoints_results[0].boxes:
            x_min_kp, y_min_kp, x_max_kp, y_max_kp = result.xyxy[0]
            kp_x_center = (x_min_kp + x_max_kp) / 2
            kp_y_center = (y_min_kp + y_max_kp) / 2

            # # Convert keypoint coordinates to mask coordinates
            # kp_x_resized = int(kp_x_center + x_min)
            # kp_y_resized = int(kp_y_center + y_min)
            # Check if keypoint is inside the mask
            # if 0 <= kp_x_center < resized_mask.shape[1] and 0 <= kp_y_center < resized_mask.shape[0]:
            #     if resized_mask[int(kp_y_center), int(kp_x_center)] > 0:
            #         keypoints.append((kp_x_center, kp_y_center))
            keypoints.append((kp_x_center, kp_y_center))
        if len(keypoints) >= 4:
            # Sort keypoints by y-axis
            keypoints.sort(key=lambda kp: kp[1])

            # Shoulders
            shoulders = keypoints[:2]
            shoulders.sort(key=lambda kp: kp[0])

            # Chest
            chest = keypoints[2]

            # Distance filtering
            if shoulders[0][0] < chest[0] < shoulders[1][0]:
                remaining_keypoints = keypoints[3:]
                belly_button = None

                threshold_middle = abs(shoulders[0][0] - shoulders[1][0]) / 3

                for kp in remaining_keypoints:
                    # Calculate distances
                    distance_chest_shoulders = abs(shoulders[1][0] - shoulders[0][0])  # Horizontal distance between shoulders
                    distance_belly_chest = abs(kp[1] - chest[1])  # Vertical distance between belly button and chest

                    # Proportional Filtering
                    proportion_threshold = 0.7  # Example value
                    if distance_belly_chest > (distance_chest_shoulders* proportion_threshold):
                        if abs(kp[0] - chest[0]) < threshold_middle and shoulders[0][0] < kp[0] < shoulders[1][0]:
                            belly_button = kp
                            break

                if belly_button is not None:
                    yolo_keypoints = [((kp[0] + x_min) / width, (kp[1] + y_min) / height) for kp in [shoulders[0], shoulders[1], chest, belly_button]]
                    keypoints_str = " ".join(f"{kp[0]} {kp[1]} 2" for kp in yolo_keypoints)

                    class_index = 0  # Assuming 'person' class index is 0
                    x_center = (x_min + x_max) / 2 / width
                    y_center = (y_min + y_max) / 2 / height
                    bbox_width = (x_max - x_min) / width
                    bbox_height = (y_max - y_min) / height
                    yolo_format = f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height} {keypoints_str}"

                    # Draw keypoints on the original image
                    for kp in yolo_keypoints:
                        cv2.circle(image, (int(kp[0] * width), int(kp[1] * height)), 5, (255, 0, 0), -1)  # Blue dots
                else:
                    print("Skipping image due to incorrect keypoints alignment")
            else:
                print("Skipping image due to incorrect keypoints alignment")
        else:
            print("Skipping image due to insufficient number of keypoints detected")
    else:
        print("No mask detected.")

output_path = './test_img/img_with_keypoints.jpg'
cv2.imwrite(output_path, image)
cv2.imshow("Image with Keypoints", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
