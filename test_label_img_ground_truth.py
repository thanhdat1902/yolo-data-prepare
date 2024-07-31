import os
import cv2

def load_keypoint_annotations(file_path):
    with open(file_path, 'r') as f:
        annotations = f.readlines()
    keypoints_data = []
    for annotation in annotations:
        parts = list(map(float, annotation.strip().split()))
        class_id = parts[0]
        bbox = parts[1:5]
        keypoints = parts[5:]
        keypoints_data.append((class_id, bbox, keypoints))
    return keypoints_data

def process_image_with_keypoints(image_path, annotation_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    keypoints_data = load_keypoint_annotations(annotation_path)

    for data in keypoints_data:
        class_id, bbox, keypoints = data
        x_center, y_center, width_norm, height_norm = bbox
        x_center *= width
        y_center *= height
        box_width = width_norm * width
        box_height = height_norm * height
        x_min = int(x_center - box_width / 2)
        y_min = int(y_center - box_height / 2)
        x_max = int(x_center + box_width / 2)
        y_max = int(y_center + box_height / 2)
        
        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw keypoints
        for i in range(0, len(keypoints), 3):
            kp_x = int(keypoints[i] * width)
            kp_y = int(keypoints[i + 1] * height)
            cv2.circle(image, (kp_x, kp_y), 5, (255, 0, 0), -1)  # Blue dot

    return image

# Set paths to your dataset directory
images_dir = './images/P14-1'
annotations_dir = './annotations/P14-1'

# images_dir = './test_image_keypoints/img'
# annotations_dir = './test_image_keypoints/label'

# images_dir = './test_img'
# annotations_dir = './test_img'

# Create a window to display the images
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# Process and show each image in real-time
for image_file in sorted(os.listdir(images_dir)):
    if image_file.endswith(('.jpg', '.png', '.jpeg', ".PNG")):
        image_path = os.path.join(images_dir, image_file)
        annotation_path = os.path.join(annotations_dir, os.path.splitext(image_file)[0] + '.txt')
        if os.path.exists(annotation_path):
            image_with_keypoints = process_image_with_keypoints(image_path, annotation_path)
            
            # Resize image if necessary to fit window
            resized_image = cv2.resize(image_with_keypoints, (640, 480))
            
            # Display the image
            cv2.imshow('Video', resized_image)
            print(image_path)
            # Video testing
            if cv2.waitKey(1000) & 0xFF == 27:  # Esc key to exit
                break

            # Single image testing
            # if cv2.waitKey(10) & 0xFF == 27:  # Esc key to exit
            #     break

# Release all windows
cv2.destroyAllWindows()
