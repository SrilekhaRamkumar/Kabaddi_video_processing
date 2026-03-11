import cv2

# Path to your video file
video_path = "Videos/raid1.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()

if ret:
    # Show the frame in a window
    cv2.imshow("First Frame", frame)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()
else:
    print("Could not read the video.")

# Release the video capture object
cap.release()