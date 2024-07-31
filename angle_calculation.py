from ultralytics import YOLO
import cv2
import numpy as np

def calculate_angle(p1, p2):
    vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    vertical = np.array([0, 1])
    # Dot product and magnitude
    dot_product = np.dot(vector, vertical)
    magnitude = np.linalg.norm(vector) * np.linalg.norm(vertical)
    angle_rad = np.arccos(dot_product / magnitude)
    angle_sign = -np.sign(vector[0])
    angle_deg = angle_sign * np.degrees(angle_rad)
    return angle_deg

class PoseEstimation:
    def __init__(self, video_name):
        self.model = YOLO('yolov8n-pose.pt')
        self.video_path = video_name
        self.upper_body_keypoints = [5, 6, 11, 12]

    def analyze_pose(self):
        color = (255, 255, 0)
        cv2.namedWindow("KeyPoints on Video", cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy.cpu().numpy()[0]
                points = {}
                for index in self.upper_body_keypoints:
                    if index < len(keypoints):
                        x, y = int(keypoints[index, 0]), int(keypoints[index, 1])
                        points[index] = (x, y)
                        cv2.circle(frame, (x, y), 5, color, -1)

                # Calculate midpoints
                if 5 in points and 6 in points and 11 in points and 12 in points:
                    midpoint_shoulder = ((points[5][0] + points[6][0]) // 2, (points[5][1] + points[6][1]) // 2)
                    midpoint_hip = ((points[11][0] + points[12][0]) // 2, (points[11][1] + points[12][1]) // 2)
                    cv2.line(frame, midpoint_shoulder, midpoint_hip, color, 2)
                    angle = calculate_angle(midpoint_shoulder, midpoint_hip)
                    cv2.putText(frame, f"Angle: {angle:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("KeyPoints on Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def run_analyze_pose():
    pe = PoseEstimation('p13_front_3.mp4')
    pe.analyze_pose()

if __name__ == '__main__':
    run_analyze_pose()
