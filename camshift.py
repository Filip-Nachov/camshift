import cv2
import numpy as np

# Function to perform CAMshift object tracking
def camshift_tracking(video_path):
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = cap.read()

    # Set up initial tracking window
    x, y, w, h = 200, 200, 100, 100
    track_window = (x, y, w, h)

    # Convert initial window to HSV color space
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Calculate histogram of the ROI
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Set termination criteria
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate back projection
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw the tracking result
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        cv2.imshow('CAMShift Object Tracking', img)

        # Press 'q' to exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'test2.mp4'  # Provide the path to your video file
camshift_tracking(video_path)
