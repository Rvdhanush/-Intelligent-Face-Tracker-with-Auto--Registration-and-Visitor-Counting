"""
ByteTrack implementation for face tracking.
Based on the ByteTrack algorithm for multi-object tracking.
"""

import numpy as np
from collections import OrderedDict
import cv2
from filterpy.kalman import KalmanFilter
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
import lap

@dataclass
class TrackState:
    """Track state enumeration."""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

@dataclass
class Detection:
    """Detection data structure."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    face_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None

class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes."""
    
    count = 0
    
    def __init__(self, bbox: np.ndarray, score: float):
        """Initialize Kalman filter for bbox tracking."""
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.score = score
        
    def _convert_bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        """Convert bounding box to measurement format."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x: np.ndarray) -> np.ndarray:
        """Convert state to bounding box format."""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([
            x[0] - w / 2.0,
            x[1] - h / 2.0,
            x[0] + w / 2.0,
            x[1] + h / 2.0
        ]).flatten()
    
    def update(self, bbox: np.ndarray, score: float):
        """Update tracker with new detection."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.score = score
        self.kf.update(self._convert_bbox_to_z(bbox))
    
    def predict(self) -> np.ndarray:
        """Predict next state and return predicted bounding box."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self) -> np.ndarray:
        """Get current bounding box estimate."""
        return self._convert_x_to_bbox(self.kf.x)

class STrack:
    """Single track for ByteTrack."""
    
    shared_kalman = KalmanBoxTracker
    track_id = 0
    
    def __init__(self, detection: Detection, frame_id: int):
        """Initialize track."""
        # Convert detection bbox to [x1, y1, x2, y2] format
        if len(detection.bbox) == 4 and detection.bbox[2] > detection.bbox[0]:
            # Already in [x1, y1, x2, y2] format
            bbox = detection.bbox
        else:
            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = detection.bbox
            bbox = np.array([x, y, x + w, y + h])
        
        self.kalman_filter = KalmanBoxTracker(bbox, detection.score)
        self.track_id = STrack.track_id
        STrack.track_id += 1
        
        self.state = TrackState.Tracked
        self.is_activated = False
        self.score = detection.score
        self.tracklet_len = 0
        self.face_id = detection.face_id
        self.embedding = detection.embedding
        self.frame_id = frame_id
        self.start_frame = frame_id
        
    def predict(self):
        """Predict next state."""
        mean_state = self.kalman_filter.predict()
        self.mean = mean_state
    
    def activate(self, frame_id: int):
        """Activate track."""
        self.track_id = STrack.track_id
        STrack.track_id += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def re_activate(self, new_track, frame_id: int):
        """Re-activate lost track."""
        self.kalman_filter.update(new_track.kalman_filter.get_state(), new_track.score)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        if new_track.face_id:
            self.face_id = new_track.face_id
        if new_track.embedding is not None:
            self.embedding = new_track.embedding
    
    def update(self, new_track, frame_id: int):
        """Update track with new detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        bbox = new_track.kalman_filter.get_state()
        self.kalman_filter.update(bbox, new_track.score)
        
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        
        if new_track.face_id:
            self.face_id = new_track.face_id
        if new_track.embedding is not None:
            self.embedding = new_track.embedding
    
    def mark_lost(self):
        """Mark track as lost."""
        self.state = TrackState.Lost
    
    def mark_removed(self):
        """Mark track as removed."""
        self.state = TrackState.Removed
    
    @property
    def tlbr(self) -> np.ndarray:
        """Get track bounding box in [x1, y1, x2, y2] format."""
        if hasattr(self, 'mean'):
            return self.mean
        else:
            return self.kalman_filter.get_state()
    
    @property
    def tlwh(self) -> np.ndarray:
        """Get track bounding box in [x, y, w, h] format."""
        ret = self.tlbr.copy()
        ret[2:] -= ret[:2]
        return ret

def iou_distance(atracks: List[STrack], btracks: List[STrack]) -> np.ndarray:
    """Calculate IoU distance matrix between tracks."""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    
    atlbrs = np.array([track.tlbr for track in atracks])
    btlbrs = np.array([track.tlbr for track in btracks])
    
    ious = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    
    for i, atlbr in enumerate(atlbrs):
        for j, btlbr in enumerate(btlbrs):
            ious[i, j] = calculate_iou(atlbr, btlbr)
    
    return 1 - ious  # Convert IoU to distance

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def linear_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve linear assignment problem."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    
    return matches, unmatched_a, unmatched_b

class ByteTracker:
    """ByteTrack multi-object tracker."""
    
    def __init__(self, frame_rate: int = 30, track_thresh: float = 0.5, 
                 track_buffer: int = 30, match_thresh: float = 0.8):
        """Initialize ByteTracker."""
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.frame_id = 0
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        
        logging.info("ByteTracker initialized")
    
    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float, Optional[str], Optional[np.ndarray]]]) -> List[STrack]:
        """Update tracker with new detections."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Convert detections to Detection objects
        detection_objects = []
        for bbox, score, face_id, embedding in detections:
            # Convert (x, y, w, h) to (x1, y1, x2, y2)
            x, y, w, h = bbox
            det_bbox = np.array([x, y, x + w, y + h])
            detection_objects.append(Detection(det_bbox, score, face_id, embedding))
        
        # Separate high and low score detections
        high_score_dets = [det for det in detection_objects if det.score >= self.track_thresh]
        low_score_dets = [det for det in detection_objects if det.score < self.track_thresh]
        
        # Create tracks from high score detections
        if len(high_score_dets) > 0:
            detections_high = [STrack(det, self.frame_id) for det in high_score_dets]
        else:
            detections_high = []
        
        # Predict current tracks
        for track in self.tracked_stracks:
            track.predict()
        
        # First association with high score detections
        strack_pool = self.tracked_stracks + self.lost_stracks
        
        if len(strack_pool) > 0 and len(detections_high) > 0:
            dists = iou_distance(strack_pool, detections_high)
            matches, u_track, u_detection = linear_assignment(dists)
            
            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections_high[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id)
                    refind_stracks.append(track)
        else:
            u_track = list(range(len(strack_pool)))
            u_detection = list(range(len(detections_high)))
        
        # Second association with low score detections
        if len(low_score_dets) > 0:
            detections_low = [STrack(det, self.frame_id) for det in low_score_dets]
            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            
            if len(r_tracked_stracks) > 0 and len(detections_low) > 0:
                dists = iou_distance(r_tracked_stracks, detections_low)
                matches, u_track_remain, u_detection_low = linear_assignment(dists)
                
                for itracked, idet in matches:
                    track = r_tracked_stracks[itracked]
                    det = detections_low[idet]
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
        
        # Handle unmatched tracks
        for it in u_track:
            track = strack_pool[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        # Create new tracks from unmatched high score detections
        unmatched_dets = [detections_high[i] for i in u_detection]
        for det in unmatched_dets:
            det.activate(self.frame_id)
            activated_stracks.append(det)
        
        # Remove old lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
        
        # Update track lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks.extend(activated_stracks)
        self.tracked_stracks.extend(refind_stracks)
        
        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]
        self.lost_stracks.extend(lost_stracks)
        
        self.removed_stracks.extend(removed_stracks)
        
        return self.tracked_stracks
    
    def get_active_tracks(self) -> List[STrack]:
        """Get currently active tracks."""
        return [track for track in self.tracked_stracks if track.is_activated]
