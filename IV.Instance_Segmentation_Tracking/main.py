import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("IV.Instance_Segmentation_Tracking/data/crowd.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("IV.Instance_Segmentation_Tracking/isegment_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize instance segmentation object
isegment = solutions.InstanceSegmentation(
    show=True,  # display the output
    model="yolo26n-seg.pt",  # model="yolo26n-seg.pt" for object segmentation using YOLO26.
    classes=[0],  # segment specific classes, e.g., person and car with the pretrained model.
    imgsz=1280,
    conf=0.2
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = isegment(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows