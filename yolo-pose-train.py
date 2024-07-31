from ultralytics import YOLO
# Display model information (optional)
if __name__ == '__main__':
    model = YOLO("yolov8x-pose-p6.yaml")  # build a new model from YAML
    model = YOLO("yolov8x-pose-p6.pt")  # load a pretrained model (recommended for training)

    epochs = 30  # Initial number of epochs
    batch_size = 8  # Adjust based on your GPU memory
    patience = 10  # Early stopping patience

    # Train the model
    results = model.train(data="./data_mobile.yaml", epochs=epochs, imgsz=640, batch=batch_size, patience=patience)
