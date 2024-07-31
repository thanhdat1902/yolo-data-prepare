import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, average_precision_score

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Function to load YOLO annotations from .txt files
def load_yolo_annotations(annotation_folder):
    annotations = {}
    for filename in os.listdir(annotation_folder):
        if filename.endswith('.txt'):
            image_id = filename.replace('.txt', '.PNG')
            filepath = os.path.join(annotation_folder, filename)
            keypoints = []
            with open(filepath, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) > 5:
                        # Extract keypoints and visibility
                        keypoints.extend([(float(parts[i]), float(parts[i+1]), float(parts[i+2])) for i in range(5, len(parts), 3)])
            annotations[image_id] = keypoints
    return annotations

# Function to evaluate a model
def evaluate_model(model, dataset, image_folder):
    predictions = {}
    print("Evaluating model...")
    for image_id in tqdm(dataset.keys(), desc="Processing Images"):
        image_path = os.path.join(image_folder, image_id)
        image = cv2.imread(image_path)
        results = model(image)
        keypoints = results[0].keypoints.xy.cpu().numpy()[0]
        predictions[image_id] = keypoints
    return predictions

# Function to compute Precision, Recall, and mAP
def compute_metrics(predictions, ground_truth):
    y_true = []
    y_pred = []
    
    print("Computing metrics...")
    for image_id in tqdm(predictions.keys(), desc="Processing Predictions"):
        pred_keypoints = predictions[image_id]
        true_keypoints = ground_truth.get(image_id, [])
        
        # Flatten keypoints for comparison
        y_true.extend([kp for kp in true_keypoints])
        y_pred.extend([kp for kp in pred_keypoints])
    
    # Example metric calculations
    # Adjust thresholds and calculations as needed for your specific use case
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    average_precision = average_precision_score(y_true, y_pred, average='macro')
    
    return precision, recall, average_precision

def main():
    # Define relative paths
    image_folder = os.path.join(BASE_DIR, "datasets_org/test/images/")
    annotation_folder = os.path.join(BASE_DIR, "datasets_org/test/labels/")
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_yolo_annotations(annotation_folder)
    
    # Load models
    custom_model_path = "./best.pt"
    pretrained_model_path = "./yolov8x-pose-p6.pt"
    
    print("Loading models...")
    custom_model = YOLO(custom_model_path)
    pretrained_model = YOLO(pretrained_model_path)
    
    # Evaluate both models
    print("Evaluating custom model...")
    custom_predictions = evaluate_model(custom_model, dataset, image_folder)
    
    print("Evaluating pretrained model...")
    pretrained_predictions = evaluate_model(pretrained_model, dataset, image_folder)
    
    # Compute metrics
    print("Computing metrics for custom model...")
    custom_precision, custom_recall, custom_ap = compute_metrics(custom_predictions, dataset)
    
    print("Computing metrics for pretrained model...")
    pretrained_precision, pretrained_recall, pretrained_ap = compute_metrics(pretrained_predictions, dataset)
    
    # Print results
    print(f"Custom Model - Precision: {custom_precision}, Recall: {custom_recall}, mAP: {custom_ap}")
    print(f"Pretrained Model - Precision: {pretrained_precision}, Recall: {pretrained_recall}, mAP: {pretrained_ap}")

if __name__ == "__main__":
    main()
