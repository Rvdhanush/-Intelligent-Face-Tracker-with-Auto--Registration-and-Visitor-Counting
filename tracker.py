"""
Face tracking module using DeepSort/ByteTrack.
Handles face tracking across video frames and manages entry/exit events.
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import time
from dataclasses import dataclass

@dataclass
class TrackedFace:
    """Data class for tracked face information."""
    track_id: int
    face_id: Optional[str]
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    embedding: Optional[np.ndarray]
    last_seen: float
    entry_logged: bool
    exit_logged: bool
    frames_missing: int
    total_detections: int

class FaceTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0, 
                 entry_exit_buffer: int = 5):
        """
        Initialize face tracker.
        
        Args:
            max_disappeared: Maximum frames a face can be missing before considered gone
            max_distance: Maximum distance for associating detections with tracks
            entry_exit_buffer: Minimum frames to confirm entry/exit
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.entry_exit_buffer = entry_exit_buffer
        
        # Tracking state
        self.tracks: Dict[int, TrackedFace] = {}
        self.next_track_id = 1
        self.frame_count = 0
        
        # Entry/Exit detection
        self.frame_width = None
        self.frame_height = None
        self.entry_zones = None
        self.exit_zones = None
        
        # History for better tracking
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        
        logging.info("Face tracker initialized")
    
    def set_frame_dimensions(self, width: int, height: int):
        """Set frame dimensions for entry/exit zone calculation."""
        self.frame_width = width
        self.frame_height = height
        
        # Define entry/exit zones (edges of the frame)
        margin = 50  # pixels from edge
        self.entry_zones = {
            'left': (0, 0, margin, height),
            'right': (width - margin, 0, margin, height),
            'top': (0, 0, width, margin),
            'bottom': (0, height - margin, width, margin)
        }
        
        logging.info(f"Set frame dimensions: {width}x{height}")
    
    def calculate_distance(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between two bounding boxes (center points)."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to (x1, y1, x2, y2) format
        box1 = (x1, y1, x1 + w1, y1 + h1)
        box2 = (x2, y2, x2 + w2, y2 + h2)
        
        # Calculate intersection
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def is_in_entry_zone(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if bounding box is in any entry zone."""
        if not self.entry_zones:
            return False
        
        x, y, w, h = bbox
        center_x, center_y = x + w/2, y + h/2
        
        for zone_name, (zx, zy, zw, zh) in self.entry_zones.items():
            if zx <= center_x <= zx + zw and zy <= center_y <= zy + zh:
                return True
        
        return False
    
    def predict_next_position(self, track_id: int) -> Optional[Tuple[int, int, int, int]]:
        """Predict next position based on track history."""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return None
        
        history = list(self.track_history[track_id])
        
        # Simple linear prediction based on last two positions
        last_bbox = history[-1]
        prev_bbox = history[-2]
        
        dx = last_bbox[0] - prev_bbox[0]
        dy = last_bbox[1] - prev_bbox[1]
        
        predicted_x = last_bbox[0] + dx
        predicted_y = last_bbox[1] + dy
        
        return (predicted_x, predicted_y, last_bbox[2], last_bbox[3])
    
    def update_tracks(self, detections: List[Tuple[Tuple[int, int, int, int], float, Optional[str], Optional[np.ndarray]]]) -> List[TrackedFace]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of (bbox, confidence, face_id, embedding) tuples
            
        Returns:
            List of current tracked faces
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Convert detections to a more manageable format
        detection_data = []
        for bbox, confidence, face_id, embedding in detections:
            detection_data.append({
                'bbox': bbox,
                'confidence': confidence,
                'face_id': face_id,
                'embedding': embedding
            })
        
        # Association matrix for Hungarian algorithm (simplified version)
        if self.tracks and detection_data:
            associations = self._associate_detections_to_tracks(detection_data)
        else:
            associations = []
        
        # Update existing tracks
        updated_track_ids = set()
        
        for detection_idx, track_id in associations:
            detection = detection_data[detection_idx]
            track = self.tracks[track_id]
            
            # Update track
            track.bbox = detection['bbox']
            track.confidence = detection['confidence']
            track.last_seen = current_time
            track.frames_missing = 0
            track.total_detections += 1
            
            # Update face_id if we have a better recognition
            if detection['face_id'] and (not track.face_id or track.confidence < detection['confidence']):
                track.face_id = detection['face_id']
                track.embedding = detection['embedding']
            
            # Update history
            self.track_history[track_id].append(detection['bbox'])
            
            updated_track_ids.add(track_id)
        
        # Create new tracks for unassociated detections
        associated_detections = {assoc[0] for assoc in associations}
        for i, detection in enumerate(detection_data):
            if i not in associated_detections:
                self._create_new_track(detection, current_time)
        
        # Update missing tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in updated_track_ids:
                track.frames_missing += 1
                
                # Mark for removal if missing too long
                if track.frames_missing > self.max_disappeared:
                    tracks_to_remove.append(track_id)
        
        # Remove old tracks
        for track_id in tracks_to_remove:
            self._remove_track(track_id)
        
        return list(self.tracks.values())
    
    def _associate_detections_to_tracks(self, detections: List[Dict]) -> List[Tuple[int, int]]:
        """Associate detections to existing tracks using distance and IoU."""
        if not self.tracks or not detections:
            return []
        
        associations = []
        used_detections = set()
        used_tracks = set()
        
        # Calculate cost matrix (distance + IoU)
        costs = []
        track_ids = list(self.tracks.keys())
        
        for track_id in track_ids:
            track = self.tracks[track_id]
            track_costs = []
            
            for detection in detections:
                # Calculate distance cost
                distance = self.calculate_distance(track.bbox, detection['bbox'])
                distance_cost = distance / self.max_distance
                
                # Calculate IoU cost (1 - IoU for cost)
                iou = self.calculate_iou(track.bbox, detection['bbox'])
                iou_cost = 1.0 - iou
                
                # Combined cost - prioritize IoU over distance for better tracking
                total_cost = 0.3 * distance_cost + 0.7 * iou_cost
                track_costs.append(total_cost)
            
            costs.append(track_costs)
        
        # Simple greedy assignment (for production, use Hungarian algorithm)
        while True:
            min_cost = float('inf')
            best_track_idx = -1
            best_detection_idx = -1
            
            for track_idx, track_id in enumerate(track_ids):
                if track_idx in used_tracks:
                    continue
                
                for detection_idx in range(len(detections)):
                    if detection_idx in used_detections:
                        continue
                    
                    cost = costs[track_idx][detection_idx]
                    if cost < min_cost and cost < 0.8:  # Much more lenient threshold
                        min_cost = cost
                        best_track_idx = track_idx
                        best_detection_idx = detection_idx
            
            if best_track_idx == -1:
                break
            
            # Make association
            track_id = track_ids[best_track_idx]
            associations.append((best_detection_idx, track_id))
            used_detections.add(best_detection_idx)
            used_tracks.add(best_track_idx)
        
        return associations
    
    def _create_new_track(self, detection: Dict, current_time: float):
        """Create a new track from detection."""
        track = TrackedFace(
            track_id=self.next_track_id,
            face_id=detection['face_id'],
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            embedding=detection['embedding'],
            last_seen=current_time,
            entry_logged=False,
            exit_logged=False,
            frames_missing=0,
            total_detections=1
        )
        
        self.tracks[self.next_track_id] = track
        self.track_history[self.next_track_id].append(detection['bbox'])
        
        logging.info(f"Created new track {self.next_track_id} for face {detection['face_id']}")
        self.next_track_id += 1
    
    def _remove_track(self, track_id: int):
        """Remove a track and clean up."""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            logging.info(f"Removing track {track_id} for face {track.face_id}")
            
            del self.tracks[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]
    
    def get_entry_exit_events(self) -> List[Tuple[str, str, TrackedFace]]:
        """
        Detect entry and exit events based on track positions and history.
        
        Returns:
            List of (event_type, face_id, track) tuples
        """
        events = []
        
        for track_id, track in self.tracks.items():
            # Check for entry events - simplified logic for better detection
            if not track.entry_logged and track.total_detections >= 3 and track.face_id:
                track.entry_logged = True
                events.append(('entry', track.face_id, track))
                logging.info(f"Entry event detected for face {track.face_id}")
            
            # Check for exit events (when track is about to be removed)
            if track.frames_missing >= self.max_disappeared - 10 and not track.exit_logged:
                if track.entry_logged and track.face_id:  # Only log exit if entry was logged
                    track.exit_logged = True
                    events.append(('exit', track.face_id, track))
                    logging.info(f"Exit event detected for face {track.face_id}")
        
        return events
    
    def get_active_faces(self) -> List[str]:
        """Get list of currently active face IDs."""
        active_faces = []
        for track in self.tracks.values():
            if track.face_id and track.frames_missing < 5:
                active_faces.append(track.face_id)
        return active_faces
    
    def get_track_by_face_id(self, face_id: str) -> Optional[TrackedFace]:
        """Get track information by face ID."""
        for track in self.tracks.values():
            if track.face_id == face_id:
                return track
        return None
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.track_history.clear()
        self.next_track_id = 1
        self.frame_count = 0
        logging.info("Tracker reset")
    
    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics."""
        active_tracks = len([t for t in self.tracks.values() if t.frames_missing < 5])
        total_tracks = len(self.tracks)
        
        return {
            'active_tracks': active_tracks,
            'total_tracks': total_tracks,
            'frame_count': self.frame_count,
            'next_track_id': self.next_track_id
        }
