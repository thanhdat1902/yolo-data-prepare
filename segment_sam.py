import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO models for detection and segmentation
model = YOLO("yolov9e.pt")
model1 = YOLO("./best.pt")
segmentation_model = YOLO("yolov8n-seg.pt")  # Assuming YOLOv8n-seg is the segmentation model

# Read the image
image = cv2.imread("test_img/frame_003634.PNG")
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

    # Ensure the mask data is present and correct
    if seg_results[0].masks is not None and len(seg_results[0].masks) > 0:
        mask_data = seg_results[0].masks.data  # Raw mask data
        mask = mask_data[0].cpu().numpy()  # Convert to NumPy array
        
        # Debug prints to check the mask type and content
        print("Mask type:", type(mask))
        print("Mask shape:", mask.shape)
        print("Mask dtype:", mask.dtype)

        # Ensure mask is a binary mask
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)  # Threshold and convert to uint8

        # Resize the mask to the original bounding box size
        resized_mask = cv2.resize(mask, (x_max - x_min, y_max - y_min))
        print(resized_mask)
        # Debug information: check if the resized mask is correctly formed
        print("Resized mask shape:", resized_mask.shape)
        print("Non-zero values in resized mask:", np.sum(resized_mask > 0))

        # Overlay the mask on the original image (color the mask area in red)
        mask_colored = np.zeros_like(image[y_min:y_max, x_min:x_max])
        mask_colored[resized_mask > 0] = [0, 0, 255]  # Red color for the mask

        # Blend the original cropped image with the mask overlay
        blended_image = cv2.addWeighted(cropped_image, 0.7, mask_colored, 0.3, 0)

        # Place the blended image back into the original image
        image[y_min:y_max, x_min:x_max] = blended_image

        # Save and show the output image
        output_path = './test_img/img_with_mask.jpg'
        cv2.imwrite(output_path, image)
        cv2.imshow("Image with Mask", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("No mask detected.")
else:
    print("No person detected.")

output_path = './test_img/img.jpg'
cv2.imwrite(output_path, image)
