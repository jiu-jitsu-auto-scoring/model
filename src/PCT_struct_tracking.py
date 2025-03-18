# # Create and activate conda environment
# # conda create -n PCT python=3.8 -y
# # conda activate PCT

# # Install dependencies from requirements.txt
# # Clone the repository and install dependencies
# !git clone https://github.com/IDEA-Research/Pose-as-Compositional-Tokens.git
# %cd Pose-as-Compositional-Tokens
# !pip install -r requirements.txt

# # Additional dependencies for our pipeline
# !pip install opencv-python scipy matplotlib


import os
import requests
import shutil
import sys
import numpy as np
import cv2
import torch


# Create necessary directories if they don't exist
os.makedirs('data/jujitsu/videos', exist_ok=True)
os.makedirs('data/jujitsu/annotations', exist_ok=True)
os.makedirs('data/jujitsu/detection_results', exist_ok=True)
os.makedirs('weights/pct', exist_ok=True)

# Here, you'd place your jujitsu videos in data/jujitsu/videos

def download_model(url, save_path):
    """
    Download a model from URL
    
    Args:
        url: URL to download from
        save_path: Path to save the model
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Downloading {url} to {save_path}")
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    print(f"Downloaded {save_path}")

# Download pre-trained models
# Note: Replace these URLs with the actual URLs from the GitHub repo
pct_model_url = "https://example.com/path/to/pct_base.pth"  # Replace with actual URL
mmdet_model_url = "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth"

download_model(pct_model_url, "weights/pct/swin_base.pth")
download_model(mmdet_model_url, "weights/mmdet/cascade_rcnn_x101_64x4d_fpn.pth")


class PCTPoseEstimator:
    def __init__(self, det_config, det_checkpoint, pose_config, pose_checkpoint, device='cuda:0'):
        """
        Initialize the pose estimator using PCT
        
        Args:
            det_config: Path to detection model config
            det_checkpoint: Path to detection model checkpoint
            pose_config: Path to pose model config
            pose_checkpoint: Path to pose model checkpoint
            device: Device to run inference on
        """
        self.device = device
        
        # Initialize detection model
        self.det_model = init_detector(
            det_config, det_checkpoint, device=device
        )
        
        # Initialize pose model
        from mmpose.apis import init_pose_model, inference_pose_model
        self.pose_model = init_pose_model(
            pose_config, pose_checkpoint, device=device
        )
        
        # Get keypoint colors
        self.palette = np.array([
            [255, 128, 0], [255, 153, 51], [255, 178, 102],
            [230, 230, 0], [255, 153, 255], [153, 204, 255],
            [255, 102, 255], [255, 51, 255], [102, 178, 255],
            [51, 153, 255], [255, 153, 153], [255, 102, 102],
            [255, 51, 51], [153, 255, 153], [102, 255, 102],
            [51, 255, 51], [0, 255, 0]
        ])
        
        # Define skeleton connections for visualization
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        # Link colors for visualization
        self.link_colors = [
            (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
            (0, 255, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
            (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
            (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
            (0, 0, 255), (0, 0, 255), (0, 0, 255)
        ]
    
    def detect_and_estimate(self, image):
        """
        Detect humans and estimate poses in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detected poses
        """
        # Detect persons
        det_results = inference_detector(self.det_model, image)
        
        # Keep only person class (typically class 0 in COCO)
        person_results = det_results[0]
        
        # Filter detections with confidence > 0.5
        person_results = [
            bbox for bbox in person_results if bbox[4] > 0.5
        ]
        
        # Convert to the format required by inference_pose_model
        person_results = [
            {'bbox': bbox} for bbox in person_results
        ]
        
        # Estimate poses
        pose_results = []
        from mmpose.apis import inference_pose_model
        
        for person_result in person_results:
            pose_result = inference_pose_model(
                self.pose_model, image, person_result
            )
            pose_results.append(pose_result)
        
        return pose_results
    
    def process_poses(self, pose_results):
        """
        Process pose results to standardized format
        
        Args:
            pose_results: Raw pose estimation results
            
        Returns:
            List of poses in standardized format
        """
        standardized_poses = []
        
        for pose_result in pose_results:
            keypoints = pose_result['keypoints']
            score = pose_result['bbox'][4] if 'bbox' in pose_result else 0.0
            
            # Convert to standard format
            pose = {
                'keypoints': keypoints,  # Shape: [17, 3] - x, y, confidence
                'score': score
            }
            standardized_poses.append(pose)
        
        return standardized_poses
    
    def visualize_poses(self, image, pose_results):
        """
        Visualize poses on image
        
        Args:
            image: Input image
            pose_results: Pose estimation results
            
        Returns:
            Image with visualized poses
        """
        vis_image = image.copy()
        
        for pose_result in pose_results:
            keypoints = pose_result['keypoints']
            
            # Draw keypoints
            for kpt_idx, kpt in enumerate(keypoints):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                if kpt_score > 0.3:
                    color = tuple(map(int, self.palette[kpt_idx % len(self.palette)]))
                    cv2.circle(vis_image, (x_coord, y_coord), 5, color, -1)
            
            # Draw skeleton
            for sk_idx, sk in enumerate(self.skeleton):
                pos1 = (int(keypoints[sk[0]-1][0]), int(keypoints[sk[0]-1][1]))
                pos2 = (int(keypoints[sk[1]-1][0]), int(keypoints[sk[1]-1][1]))
                
                if (keypoints[sk[0]-1][2] > 0.3 and keypoints[sk[1]-1][2] > 0.3):
                    color = self.link_colors[sk_idx % len(self.link_colors)]
                    cv2.line(vis_image, pos1, pos2, color, thickness=2)
        
        return vis_image
    

import numpy as np
from scipy.optimize import linear_sum_assignment

class PoseTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize pose tracker
        
        Args:
            max_age: Maximum number of frames to keep a track alive without matching
            min_hits: Minimum number of hits to consider a track confirmed
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        self.next_id = 0
    
    def update(self, poses):
        """
        Update tracks with new poses
        
        Args:
            poses: List of detected poses
            
        Returns:
            List of tracked poses with IDs
        """
        self.frame_count += 1
        
        # If no tracks exist yet, initialize with current poses
        if len(self.tracks) == 0:
            for pose in poses:
                self.tracks.append({
                    'id': self.next_id,
                    'pose': pose,
                    'age': 0,
                    'hits': 1,
                    'time_since_update': 0
                })
                self.next_id += 1
            return self.tracks
        
        # Calculate IoU between current poses and existing tracks
        iou_matrix = np.zeros((len(poses), len(self.tracks)))
        for i, pose in enumerate(poses):
            for j, track in enumerate(self.tracks):
                iou_matrix[i, j] = self._calculate_pose_similarity(pose, track['pose'])
        
        # Hungarian algorithm for matching
        pose_indices, track_indices = linear_sum_assignment(-iou_matrix)
        
        # Update matched tracks
        matched_indices = []
        for pose_idx, track_idx in zip(pose_indices, track_indices):
            if iou_matrix[pose_idx, track_idx] >= self.iou_threshold:
                self.tracks[track_idx]['pose'] = poses[pose_idx]
                self.tracks[track_idx]['hits'] += 1
                self.tracks[track_idx]['age'] = 0
                self.tracks[track_idx]['time_since_update'] = 0
                matched_indices.append(pose_idx)
            else:
                self.tracks[track_idx]['time_since_update'] += 1
        
        # Create new tracks for unmatched detections
        for i, pose in enumerate(poses):
            if i not in matched_indices:
                self.tracks.append({
                    'id': self.next_id,
                    'pose': pose,
                    'age': 0,
                    'hits': 1,
                    'time_since_update': 0
                })
                self.next_id += 1
        
        # Update track ages
        for track in self.tracks:
            track['age'] += 1
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks 
                       if track['time_since_update'] < self.max_age]
        
        # Return only confirmed tracks
        confirmed_tracks = [track for track in self.tracks 
                           if track['hits'] >= self.min_hits]
        
        return confirmed_tracks
    
    def _calculate_pose_similarity(self, pose1, pose2):
        """
        Calculate similarity between two poses using OKS (Object Keypoint Similarity)
        
        Args:
            pose1: First pose
            pose2: Second pose
            
        Returns:
            Similarity score
        """
        # Define keypoint sigmas (from COCO)
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 
            1.07, .87, .87, .89, .89
        ]) / 10.0
        
        # Get keypoints
        keypoints1 = pose1['keypoints']
        keypoints2 = pose2['keypoints']
        
        # Calculate distances
        dist = np.sum((keypoints1[:, :2] - keypoints2[:, :2]) ** 2, axis=1)
        
        # Calculate visibility
        vis = np.logical_and(keypoints1[:, 2] > 0.3, keypoints2[:, 2] > 0.3)
        
        # Get scale (use max dimension of pose bounding box)
        bbox1 = self._get_pose_bbox(pose1)
        bbox2 = self._get_pose_bbox(pose2)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        scale = np.sqrt(max(area1, area2))
        
        # Calculate OKS
        oks = np.sum(np.exp(-dist / (2 * scale * (sigmas ** 2))) * vis) / (np.sum(vis) + 1e-6)
        
        return oks
    
    def _get_pose_bbox(self, pose):
        """
        Get bounding box from pose
        
        Args:
            pose: Pose dictionary with keypoints
            
        Returns:
            Bounding box as [x1, y1, x2, y2]
        """
        keypoints = pose['keypoints']
        valid_keypoints = keypoints[keypoints[:, 2] > 0.3]
        
        if len(valid_keypoints) == 0:
            return [0, 0, 0, 0]
        
        x1 = np.min(valid_keypoints[:, 0])
        y1 = np.min(valid_keypoints[:, 1])
        x2 = np.max(valid_keypoints[:, 0])
        y2 = np.max(valid_keypoints[:, 1])
        
        # Add some padding
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, x1 - width * 0.1)
        y1 = max(0, y1 - height * 0.1)
        x2 = x2 + width * 0.1
        y2 = y2 + height * 0.1
        
        return [x1, y1, x2, y2]
    

import cv2
import numpy as np
import time
import os

def process_jujitsu_video(video_path, output_path=None, 
                         det_config='vis_tools/cascade_rcnn_x101_64x4d_fpn_coco.py',
                         det_checkpoint='weights/mmdet/cascade_rcnn_x101_64x4d_fpn.pth',
                         pose_config='configs/pct_base_classifier.py',
                         pose_checkpoint='weights/pct/swin_base.pth'):
    """
    Process jujitsu video to estimate and track poses
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video (optional)
        det_config: Path to detection model config
        det_checkpoint: Path to detection model checkpoint
        pose_config: Path to pose model config
        pose_checkpoint: Path to pose model checkpoint
        
    Returns:
        List of tracked poses across frames
    """
    # Initialize pose estimator
    pose_estimator = PCTPoseEstimator(
        det_config, det_checkpoint, pose_config, pose_checkpoint
    )
    
    # Initialize pose tracker
    tracker = PoseTracker()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize output video writer if needed
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_tracked_poses = []
    frame_idx = 0
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for faster processing (optional)
        # if frame_idx % 2 != 0:
        #     frame_idx += 1
        #     continue
        
        start_time = time.time()
        
        # Detect and estimate poses
        pose_results = pose_estimator.detect_and_estimate(frame)
        
        # Process poses to standard format
        poses = pose_estimator.process_poses(pose_results)
        
        # Track poses
        tracked_poses = tracker.update(poses)
        all_tracked_poses.append(tracked_poses)
        
        # Visualize if output is requested
        if output_path:
            vis_frame = pose_estimator.visualize_poses(frame, [t['pose'] for t in tracked_poses])
            
            # Add tracking IDs
            for track in tracked_poses:
                pose = track['pose']
                track_id = track['id']
                # Get center of pose
                keypoints = pose['keypoints']
                valid_keypoints = keypoints[keypoints[:, 2] > 0.3]
                if len(valid_keypoints) > 0:
                    center_x = int(np.mean(valid_keypoints[:, 0]))
                    center_y = int(np.mean(valid_keypoints[:, 1]))
                    # Add ID text
                    cv2.putText(vis_frame, f"ID: {track_id}", 
                               (center_x, center_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add frame number
            cv2.putText(vis_frame, f"Frame: {frame_idx}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            out.write(vis_frame)
        
        frame_idx += 1
        process_time = time.time() - start_time
        
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}, FPS: {1/process_time:.2f}")
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    
    return all_tracked_poses