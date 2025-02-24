import cv2
import logging
from pose_tracking.tracker_module import Tracker

# Set up logging to see tracker output
logging.basicConfig(level=logging.INFO)

# Configure tracker parameters (adjust as needed)
tracker_path = "models/tracker_model.pickle"  # on local (Harmony4d idk if works)
n_ids = 2  # For two competitors
img_size = (640, 480)
pose_window_len = 10
dist_thresh = 0.5
joints = ['joint1', 'joint2', 'joint3', 'joint4']  # tmp
njoints = 17
joint_conf_thresh = 0.5
dataset_update_freq = 5
classifier_update_freq = 10
vis_conf_thresh = 0.6

tracker = Tracker(tracker_path, n_ids, img_size,
                  pose_window_len, dist_thresh,
                  joints, njoints, joint_conf_thresh,
                  dataset_update_freq, classifier_update_freq,
                  vis_conf_thresh)

# Function to simulate pose detection (replace with your detector)
def detect_poses(frame):
    # This function should return a list of dicts with keys:
    # - 'keypoints': numpy array of shape (num_joints, 3)
    # - 'structure': any structure feature (if used)
    # - 'descriptors': dict of descriptors for appearance matching
    # For now, return an empty list or simulated data.
    return []

video_path = "videos/jiu_jitsu_match.mp4" #upp
cap = cv2.VideoCapture(video_path)
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get pose detections for this frame
    pose_results = detect_poses(frame)

    # Run tracking; here we use the track() method
    matched, ordered_results, updated = tracker.track(pose_results, frame_num)

    # (Optional) Visualize the results on the frame
    for result in ordered_results:
        if 'keypoints' in result:
            for (x, y, conf) in result['keypoints']:
                if conf > 0.5:  # threshold to display
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            # Optionally draw the track ID
            cv2.putText(frame, f"ID: {result.get('track_id', -1)}", 
                        (int(result['keypoints'][0, 0]), int(result['keypoints'][0, 1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()
