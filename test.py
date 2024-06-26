from ultralytics import YOLO
import torch
import cv2

if __name__ == '__main__':
    # Load a model
    model = YOLO("E:/ultralytics-main/runs/detect/train23/weights/best.pt")

    # Define path to video file
    source = "E:/ultralytics-main/2024-06-20 15-49-58.mkv"

    # Run inference on the source
    #results = model(source, stream=True)  # generator of Results objects
    # Open the video file
video_path = "E:/ultralytics-main/2024-06-25 16-42-45.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()