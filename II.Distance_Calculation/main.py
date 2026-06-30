import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("II.Distance_Calculation/data/crowd.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(
    "II.Distance_Calculation/distance_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h))

# Initialize distance calculation object
distancecalculator = solutions.DistanceCalculation(
    model="yolo26l.pt",  # path to the YOLO26 model file.
    show=True,  # display the output
    classes = [0]
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = distancecalculator(im0)

    print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows