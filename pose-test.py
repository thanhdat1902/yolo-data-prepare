from ultralytics import YOLO
import cv2

# Display model information (optional)
if __name__ == '__main__':
    model = YOLO("./models/last-hpc-keypoint.pt")  # build a new model from YAML
    # Test the model
    cap = cv2.VideoCapture("./video/Need Convert/p14_front_1.mp4")

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Loop to read frames and perform inference
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        results = model(frame)

        # Visualize the results
        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()[0]
            for keypoint in keypoints:
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Display the frame with poses
        cv2.imshow('Pose Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()