"""
Flask web application for Face Tracking System Dashboard.
Provides real-time monitoring and statistics visualization.
"""

from flask import Flask, render_template, jsonify, Response, request
from flask_socketio import SocketIO, emit
import cv2
import json
import threading
import time
from datetime import datetime, timedelta
import base64
import numpy as np
from pathlib import Path
import sqlite3
import os

from db import FaceDatabase
from simple_main import SimpleFaceTrackingSystem

app = Flask(__name__)
app.config['SECRET_KEY'] = 'face_tracking_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
face_tracking_system = None
is_processing = False
current_frame = None
current_stats = {}
processing_thread = None

class WebFaceTrackingSystem(SimpleFaceTrackingSystem):
    """Extended face tracking system for web interface."""
    
    def __init__(self, config_path="config.json"):
        super().__init__(config_path)
        self.web_stats = {
            'total_frames': 0,
            'fps': 0,
            'active_faces': 0,
            'total_detections': 0,
            'total_recognitions': 0,
            'total_entries': 0,
            'total_exits': 0,
            'processing_time': 0
        }
        
    def process_frame_web(self, frame):
        """Process frame and emit updates to web clients."""
        global current_frame, current_stats
        
        # Process frame with confidence scores visible
        processed_frame, stats = self.process_frame(frame)
        
        # Ensure confidence scores are displayed on the frame for web
        # Force display of tracking info with confidence scores
        tracked_faces = [t for t in self.tracker.tracks.values() if t.frames_missing < 20]
        processed_frame = self.draw_web_tracking_info(processed_frame, tracked_faces, stats)
        
        # Update web stats
        self.web_stats.update({
            'total_frames': self.frame_count,
            'active_faces': len([t for t in self.tracker.tracks.values() if t.frames_missing < 5]),
            'total_detections': stats.get('detections', 0),
            'total_recognitions': stats.get('recognitions', 0),
            'total_entries': stats.get('events', 0),
            'total_exits': stats.get('events', 0),
            'known_faces': len(self.known_embeddings),
            'fps': 1.0 / (np.mean(self.processing_times) if self.processing_times else 1.0)
        })
        
        # Convert frame to base64 for web display (optimized for speed)
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        current_frame = frame_base64
        current_stats = self.web_stats.copy()
        
        # Emit real-time updates
        socketio.emit('frame_update', {
            'frame': frame_base64,
            'stats': self.web_stats
        })
        
        return processed_frame, stats
    
    def draw_web_tracking_info(self, frame, tracked_faces, stats):
        """Draw stable tracking information with confidence scores for web display."""
        frame_copy = frame.copy()
        
        # Filter for active tracks with immediate display (no delay)
        stable_tracks = []
        for track in tracked_faces:
            if (track.frames_missing < 20 and  # Allow even longer tracking
                track.total_detections >= 1 and  # Show immediately (no 3-second delay)
                hasattr(track, 'confidence') and track.confidence and track.confidence > 0.2):  # Very low threshold
                stable_tracks.append(track)
        
        # Sort by confidence first (highest confidence first)
        stable_tracks.sort(key=lambda t: (t.confidence if hasattr(t, 'confidence') and t.confidence else 0), reverse=True)
        
        # ULTRA AGGRESSIVE duplicate removal using spatial clustering
        final_tracks = self._cluster_and_merge_tracks(stable_tracks)
        
        # Limit to maximum 8 tracks to prevent overcrowding
        final_tracks = final_tracks[:8]
        
        # Draw only the final, non-overlapping tracks
        for track in final_tracks:
            x, y, w, h = track.bbox
            
            # Choose color based on face status
            if track.face_id:
                color = (0, 255, 0)  # Green for recognized faces
            else:
                color = (0, 255, 255)  # Yellow for detecting faces
            
            # Draw very thick bounding box for visibility
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 6)
            
            # Create highly visible label with confidence score
            if track.face_id:
                main_label = f"ID: {track.face_id[-8:]}"
                if hasattr(track, 'confidence') and track.confidence:
                    conf_label = f"CONF: {track.confidence:.2f}"
                else:
                    conf_label = "CONF: N/A"
                status = "RECOGNIZED"
            else:
                main_label = "DETECTING"
                if hasattr(track, 'confidence') and track.confidence:
                    conf_label = f"CONF: {track.confidence:.2f}"
                else:
                    conf_label = "CONF: N/A"
                status = "PROCESSING"
            
            # Calculate text sizes with larger fonts
            main_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]
            conf_size = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            status_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Large background rectangle for better visibility
            bg_width = max(main_size[0], conf_size[0], status_size[0]) + 20
            bg_height = main_size[1] + conf_size[1] + status_size[1] + 25
            
            # Draw solid background
            cv2.rectangle(frame_copy, (x, y - bg_height - 5), 
                         (x + bg_width, y), color, -1)
            
            # Draw white border around background
            cv2.rectangle(frame_copy, (x, y - bg_height - 5), 
                         (x + bg_width, y), (255, 255, 255), 2)
            
            # Draw text with high contrast
            text_y = y - bg_height + 20
            
            # Main label (ID or DETECTING)
            cv2.putText(frame_copy, main_label, (x + 10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
            text_y += main_size[1] + 5
            
            # Confidence score
            cv2.putText(frame_copy, conf_label, (x + 10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            text_y += conf_size[1] + 5
            
            # Status
            cv2.putText(frame_copy, status, (x + 10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame_copy
    
    def _tracks_overlap(self, bbox1, bbox2, threshold=0.2):
        """Check if two bounding boxes overlap significantly."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate center points
        center1_x, center1_y = x1 + w1/2, y1 + h1/2
        center2_x, center2_y = x2 + w2/2, y2 + h2/2
        
        # Calculate distance between centers
        distance = ((center1_x - center2_x)**2 + (center1_y - center2_y)**2)**0.5
        
        # Average box size
        avg_size = (w1 + h1 + w2 + h2) / 4
        
        # Much more aggressive - if centers are close, consider overlap
        if distance < avg_size * 1.2:  # Increased from 0.8 to 1.2
            return True
        
        # Also check traditional IoU
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            iou = intersection / union if union > 0 else 0
            return iou > threshold
        
        return False
    
    def _cluster_and_merge_tracks(self, tracks):
        """Ultra aggressive clustering to merge nearby tracks into single tracks."""
        if not tracks:
            return []
        
        # Create clusters based on spatial proximity
        clusters = []
        
        for track in tracks:
            # Find if this track belongs to any existing cluster
            added_to_cluster = False
            
            for cluster in clusters:
                # Check if track is close to any track in this cluster
                for cluster_track in cluster:
                    if self._tracks_very_close(track.bbox, cluster_track.bbox):
                        cluster.append(track)
                        added_to_cluster = True
                        break
                
                if added_to_cluster:
                    break
            
            # If not added to any cluster, create new cluster
            if not added_to_cluster:
                clusters.append([track])
        
        # For each cluster, keep only the best track
        final_tracks = []
        for cluster in clusters:
            if len(cluster) == 1:
                final_tracks.append(cluster[0])
            else:
                # Choose best track from cluster (highest confidence + most detections)
                best_track = max(cluster, key=lambda t: (
                    t.confidence if hasattr(t, 'confidence') and t.confidence else 0,
                    t.total_detections,
                    -t.frames_missing  # Prefer tracks with fewer missing frames
                ))
                final_tracks.append(best_track)
        
        return final_tracks
    
    def _tracks_very_close(self, bbox1, bbox2):
        """Check if two tracks are very close (more aggressive than overlap)."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate center points
        center1_x, center1_y = x1 + w1/2, y1 + h1/2
        center2_x, center2_y = x2 + w2/2, y2 + h2/2
        
        # Calculate distance between centers
        distance = ((center1_x - center2_x)**2 + (center1_y - center2_y)**2)**0.5
        
        # Average box diagonal
        diag1 = (w1**2 + h1**2)**0.5
        diag2 = (w2**2 + h2**2)**0.5
        avg_diag = (diag1 + diag2) / 2
        
        # Very aggressive clustering - if distance is less than 1.5x average diagonal
        return distance < avg_diag * 1.5

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get current system statistics."""
    global current_stats
    
    # Get database stats
    db = FaceDatabase()
    daily_stats = db.get_daily_stats()
    recent_events = db.get_recent_events(10)
    
    # Get video files info
    video_dir = Path("Video Datasets")
    video_files = []
    if video_dir.exists():
        for video in video_dir.glob("*.mp4"):
            size_mb = video.stat().st_size / (1024*1024)
            video_files.append({
                'name': video.name,
                'size_mb': round(size_mb, 1)
            })
    
    # Get log files info
    logs_dir = Path("logs")
    log_info = {
        'entry_images': 0,
        'exit_images': 0,
        'face_images': 0,
        'log_files': 0
    }
    
    if logs_dir.exists():
        # Count images
        entries_dir = logs_dir / "images" / "entries"
        exits_dir = logs_dir / "images" / "exits"
        faces_dir = logs_dir / "images" / "faces"
        
        if entries_dir.exists():
            log_info['entry_images'] = len(list(entries_dir.rglob("*.jpg")))
        if exits_dir.exists():
            log_info['exit_images'] = len(list(exits_dir.rglob("*.jpg")))
        if faces_dir.exists():
            log_info['face_images'] = len(list(faces_dir.rglob("*.jpg")))
        
        # Count log files
        events_dir = logs_dir / "events"
        if events_dir.exists():
            log_info['log_files'] = len(list(events_dir.glob("*.log")))
    
    return jsonify({
        'system_stats': current_stats,
        'daily_stats': daily_stats,
        'recent_events': recent_events[:5],  # Last 5 events
        'video_files': video_files[:10],  # First 10 videos
        'log_info': log_info,
        'is_processing': is_processing,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/start_processing', methods=['POST'])
def start_processing():
    """Start face tracking processing."""
    global face_tracking_system, is_processing, processing_thread
    
    if is_processing:
        return jsonify({'status': 'error', 'message': 'Processing already running'})
    
    data = request.json
    video_file = data.get('video_file', 'Video Datasets/video_sample1.mp4')
    
    try:
        # Initialize system with web optimizations
        face_tracking_system = WebFaceTrackingSystem()
        face_tracking_system.config['video_source'] = video_file
        face_tracking_system.config['display']['show_video'] = False
        face_tracking_system.config['skip_frames'] = 2  # Skip more frames for stability
        # MANDATORY: Enable comprehensive logging for hackathon compliance
        face_tracking_system.config['logging']['save_images'] = True  # Save entry/exit images
        face_tracking_system.config['logging']['log_events'] = True  # Log all events
        face_tracking_system.config['logging']['log_level'] = 'INFO'  # Detailed logging
        face_tracking_system.config['logging']['base_log_dir'] = 'logs'  # Ensure correct log directory
        
        # Optimize tracking for single-person tracking (no duplicates)
        face_tracking_system.config['detection']['confidence_threshold'] = 0.5  # Balanced threshold
        face_tracking_system.config['tracking']['max_disappeared'] = 30  # Longer tracking
        face_tracking_system.config['tracking']['max_distance'] = 200.0  # Much wider distance for better association
        face_tracking_system.config['tracking']['entry_exit_buffer'] = 1  # Immediate response
        
        # CRITICAL: Initialize all components including logger
        face_tracking_system.initialize_components()
        
        # Start processing in background thread
        processing_thread = threading.Thread(target=run_processing)
        processing_thread.daemon = True
        processing_thread.start()
        
        is_processing = True
        
        return jsonify({'status': 'success', 'message': f'Started processing {video_file}'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop_processing', methods=['POST'])
def stop_processing():
    """Stop face tracking processing."""
    global is_processing
    
    is_processing = False
    
    return jsonify({'status': 'success', 'message': 'Processing stopped'})

@app.route('/api/video_files')
def get_video_files():
    """Get list of available video files."""
    video_dir = Path("Video Datasets")
    video_files = []
    
    if video_dir.exists():
        for video in video_dir.glob("*.mp4"):
            size_mb = video.stat().st_size / (1024*1024)
            video_files.append({
                'name': video.name,
                'path': str(video),
                'size_mb': round(size_mb, 1)
            })
    
    return jsonify({'video_files': video_files})

@app.route('/api/recent_images')
def get_recent_images():
    """Get recent face images."""
    logs_dir = Path("logs/images")
    recent_images = []
    
    if logs_dir.exists():
        # Get recent entry images
        entries_dir = logs_dir / "entries"
        if entries_dir.exists():
            for img_file in sorted(entries_dir.rglob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                with open(img_file, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                recent_images.append({
                    'type': 'entry',
                    'filename': img_file.name,
                    'timestamp': datetime.fromtimestamp(img_file.stat().st_mtime).isoformat(),
                    'image': img_base64
                })
        
        # Get recent exit images
        exits_dir = logs_dir / "exits"
        if exits_dir.exists():
            for img_file in sorted(exits_dir.rglob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                with open(img_file, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                recent_images.append({
                    'type': 'exit',
                    'filename': img_file.name,
                    'timestamp': datetime.fromtimestamp(img_file.stat().st_mtime).isoformat(),
                    'image': img_base64
                })
    
    # Sort by timestamp and return most recent
    recent_images.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify({'images': recent_images[:10]})

def run_processing():
    """Run face tracking processing in background."""
    global face_tracking_system, is_processing
    
    try:
        cap = face_tracking_system.setup_video_source()
        frame_count = 0
        
        while is_processing:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 2nd frame for stability while maintaining responsiveness
            if frame_count % 2 == 0:
                # Process frame with web updates
                face_tracking_system.process_frame_web(frame)
            
            # Minimal delay for smooth video playback
            time.sleep(0.02)  # ~50 FPS target
        
        cap.release()
        
    except Exception as e:
        print(f"Processing error: {e}")
    finally:
        is_processing = False

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('status', {'message': 'Connected to Face Tracking System'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting Face Tracking System Web Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🎯 Features: Real-time monitoring, Statistics, Live video feed")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
