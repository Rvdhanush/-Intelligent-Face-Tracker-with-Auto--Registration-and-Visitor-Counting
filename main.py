"""
Main module for real-time intelligent face tracking system.
Integrates detection, recognition, tracking, and logging components.
"""

import cv2
import numpy as np
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import signal
import sys

from detector import FaceDetector
from recognizer import FaceRecognizer
from tracker import FaceTracker
from logger import FaceTrackingLogger
from db import FaceDatabase

class FaceTrackingSystem:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the face tracking system."""
        self.config = self.load_config(config_path)
        self.running = False
        
        # Initialize components
        self.detector = None
        self.recognizer = None
        self.tracker = None
        self.logger = None
        self.database = None
        
        # Processing state
        self.frame_count = 0
        self.skip_frames = self.config.get('skip_frames', 2)
        self.known_embeddings = []
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.processing_times = []
        
        self.initialize_components()
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "video_source": "sample_video.mp4",
            "rtsp_url": "rtsp://example.com/stream",
            "use_rtsp": False,
            "skip_frames": 2,
            "detection": {
                "model_path": None,
                "confidence_threshold": 0.5,
                "min_face_size": 50
            },
            "recognition": {
                "model_name": "buffalo_l",
                "similarity_threshold": 0.6,
                "min_quality": 0.3
            },
            "tracking": {
                "max_disappeared": 30,
                "max_distance": 100.0,
                "entry_exit_buffer": 5
            },
            "logging": {
                "base_log_dir": "logs",
                "log_level": "INFO",
                "save_images": True,
                "cleanup_days": 30
            },
            "database": {
                "db_path": "face_tracking.db"
            },
            "display": {
                "show_video": True,
                "show_fps": True,
                "window_width": 1280,
                "window_height": 720
            }
        }
    
    def initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize detector
            self.detector = FaceDetector(
                model_path=self.config['detection'].get('model_path'),
                confidence_threshold=self.config['detection']['confidence_threshold']
            )
            
            # Initialize recognizer
            self.recognizer = FaceRecognizer(
                model_name=self.config['recognition']['model_name'],
                similarity_threshold=self.config['recognition']['similarity_threshold']
            )
            
            # Initialize tracker
            self.tracker = FaceTracker(
                max_disappeared=self.config['tracking']['max_disappeared'],
                max_distance=self.config['tracking']['max_distance'],
                entry_exit_buffer=self.config['tracking']['entry_exit_buffer']
            )
            
            # Initialize logger
            self.logger = FaceTrackingLogger(
                base_log_dir=self.config['logging']['base_log_dir'],
                log_level=self.config['logging']['log_level']
            )
            
            # Initialize database
            self.database = FaceDatabase(
                db_path=self.config['database']['db_path']
            )
            
            # Load known embeddings from database
            self.load_known_embeddings()
            
            logging.info("All components initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing components: {e}")
            raise
    
    def load_known_embeddings(self):
        """Load known face embeddings from database."""
        try:
            self.known_embeddings = self.database.get_all_embeddings()
            logging.info(f"Loaded {len(self.known_embeddings)} known face embeddings")
        except Exception as e:
            logging.error(f"Error loading known embeddings: {e}")
            self.known_embeddings = []
    
    def setup_video_source(self) -> cv2.VideoCapture:
        """Setup video capture source."""
        try:
            if self.config.get('use_rtsp', False):
                # Use RTSP stream
                source = self.config['rtsp_url']
                logging.info(f"Using RTSP source: {source}")
            else:
                # Use video file
                source = self.config['video_source']
                if not Path(source).exists():
                    logging.error(f"Video file not found: {source}")
                    raise FileNotFoundError(f"Video file not found: {source}")
                logging.info(f"Using video file: {source}")
            
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {source}")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logging.info(f"Video properties: {width}x{height} @ {fps} FPS")
            
            # Set tracker frame dimensions
            self.tracker.set_frame_dimensions(width, height)
            
            return cap
            
        except Exception as e:
            logging.error(f"Error setting up video source: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame through the entire pipeline.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (processed_frame, stats)
        """
        start_time = time.time()
        stats = {'detections': 0, 'recognitions': 0, 'new_faces': 0, 'events': 0}
        
        try:
            # Skip frames if configured
            if self.frame_count % (self.skip_frames + 1) != 0:
                self.frame_count += 1
                return frame, stats
            
            # Detect faces
            face_detections = self.detector.detect_and_crop_faces(frame)
            stats['detections'] = len(face_detections)
            
            if not face_detections:
                # Update tracker with empty detections
                self.tracker.update_tracks([])
                self.frame_count += 1
                return frame, stats
            
            # Process each detected face
            tracking_data = []
            
            for face_crop, bbox, confidence in face_detections:
                # Check face quality
                if not self.detector.is_valid_face(face_crop, self.config['detection']['min_face_size']):
                    continue
                
                if not self.recognizer.is_good_quality_face(face_crop, self.config['recognition']['min_quality']):
                    continue
                
                # Recognize face
                face_id, recognition_confidence, embedding = self.recognizer.recognize_face(
                    face_crop, self.known_embeddings
                )
                
                if face_id is None:
                    # New face - register it
                    face_id = self.recognizer.generate_face_id()
                    
                    if embedding is not None:
                        # Register in database
                        if self.database.register_face(face_id, embedding):
                            # Add to known embeddings
                            self.known_embeddings.append((face_id, embedding))
                            
                            # Log registration
                            if self.config['logging']['save_images']:
                                self.logger.log_face_registration(face_id, face_crop, embedding, confidence)
                            
                            stats['new_faces'] += 1
                            logging.info(f"Registered new face: {face_id}")
                else:
                    # Known face - update database
                    self.database.update_face_last_seen(face_id)
                    
                    # Log recognition
                    self.logger.log_face_recognition(face_id, recognition_confidence, bbox)
                    stats['recognitions'] += 1
                
                # Prepare data for tracking
                tracking_data.append((bbox, confidence, face_id, embedding))
            
            # Update tracker
            tracked_faces = self.tracker.update_tracks(tracking_data)
            
            # Check for entry/exit events
            events = self.tracker.get_entry_exit_events()
            stats['events'] = len(events)
            
            for event_type, face_id, track in events:
                # Log event to database
                image_path = None
                
                if self.config['logging']['save_images'] and track.bbox:
                    # Crop face from current frame for event logging
                    face_crop = self.detector.crop_face(frame, track.bbox)
                    if face_crop is not None:
                        if event_type == 'entry':
                            image_path = self.logger.log_face_entry(face_id, face_crop, track.bbox, track.confidence)
                        elif event_type == 'exit':
                            image_path = self.logger.log_face_exit(face_id, face_crop, track.bbox, track.confidence)
                
                # Log to database
                self.database.log_event(event_type, face_id, image_path, track.confidence, track.bbox)
                
                logging.info(f"Logged {event_type} event for face {face_id}")
            
            # Draw detections and tracking info on frame
            if self.config['display']['show_video']:
                frame = self.draw_tracking_info(frame, tracked_faces, stats)
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        self.frame_count += 1
        return frame, stats
    
    def draw_tracking_info(self, frame: np.ndarray, tracked_faces: List, stats: Dict) -> np.ndarray:
        """Draw tracking information on frame."""
        frame_copy = frame.copy()
        
        # Draw face tracks
        for track in tracked_faces:
            if track.frames_missing > 5:
                continue
            
            x, y, w, h = track.bbox
            
            # Choose color based on face status
            if track.face_id:
                color = (0, 255, 0) if track.entry_logged else (0, 255, 255)
            else:
                color = (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"ID: {track.face_id or 'Unknown'}"
            if track.confidence:
                label += f" ({track.confidence:.2f})"
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame_copy, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Label text
            cv2.putText(frame_copy, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw statistics
        if self.config['display']['show_fps']:
            self.draw_stats(frame_copy, stats)
        
        return frame_copy
    
    def draw_stats(self, frame: np.ndarray, stats: Dict):
        """Draw statistics on frame."""
        # Calculate FPS
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.fps_start_time >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
        else:
            fps = 0
        
        # Prepare stats text
        stats_text = [
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Detections: {stats['detections']}",
            f"Recognitions: {stats['recognitions']}",
            f"New Faces: {stats['new_faces']}",
            f"Events: {stats['events']}",
            f"Known Faces: {len(self.known_embeddings)}",
            f"Active Tracks: {len([t for t in self.tracker.tracks.values() if t.frames_missing < 5])}"
        ]
        
        # Draw stats background
        max_width = max([cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for text in stats_text])
        stats_height = len(stats_text) * 20 + 10
        cv2.rectangle(frame, (10, 10), (max_width + 20, stats_height), (0, 0, 0), -1)
        
        # Draw stats text
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * 20
            cv2.putText(frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Main processing loop."""
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Setup video source
            cap = self.setup_video_source()
            
            # Log system start
            self.logger.log_system_event("start", "Face tracking system started", {
                'config': self.config,
                'known_faces': len(self.known_embeddings)
            })
            
            logging.info("Starting face tracking system...")
            
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    if self.config.get('use_rtsp', False):
                        logging.warning("Lost RTSP connection, attempting to reconnect...")
                        cap.release()
                        time.sleep(5)
                        cap = self.setup_video_source()
                        continue
                    else:
                        logging.info("End of video file reached")
                        break
                
                # Process frame
                processed_frame, stats = self.process_frame(frame)
                
                # Display frame if configured
                if self.config['display']['show_video']:
                    # Resize for display if needed
                    display_width = self.config['display']['window_width']
                    display_height = self.config['display']['window_height']
                    
                    if processed_frame.shape[1] != display_width or processed_frame.shape[0] != display_height:
                        processed_frame = cv2.resize(processed_frame, (display_width, display_height))
                    
                    cv2.imshow('Face Tracking System', processed_frame)
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                    elif key == ord('s'):  # 's' for stats
                        self.print_session_stats()
                    elif key == ord('r'):  # 'r' for reset
                        self.tracker.reset()
                        logging.info("Tracker reset by user")
            
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logging.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            # Log system stop
            self.logger.log_system_event("stop", "Face tracking system stopped", {
                'total_frames': self.frame_count,
                'session_stats': self.logger.get_current_stats()
            })
            
            # Close video capture
            cv2.destroyAllWindows()
            
            # Print final stats
            self.print_session_stats()
            
            # Cleanup old logs if configured
            if self.config['logging'].get('cleanup_days'):
                self.logger.cleanup_old_logs(self.config['logging']['cleanup_days'])
                self.database.cleanup_old_data(self.config['logging']['cleanup_days'])
            
            logging.info("Cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    def print_session_stats(self):
        """Print session statistics."""
        stats = self.logger.get_current_stats()
        daily_stats = self.database.get_daily_stats()
        
        print("\n" + "="*50)
        print("SESSION STATISTICS")
        print("="*50)
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Known Faces: {len(self.known_embeddings)}")
        print(f"Active Tracks: {len(self.tracker.tracks)}")
        
        print("\nEVENT COUNTS:")
        for event_type, count in stats['session_stats'].items():
            print(f"  {event_type.title()}: {count}")
        
        print(f"\nDAILY STATISTICS ({daily_stats['date']}):")
        print(f"  Unique Visitors: {daily_stats['unique_visitors']}")
        print(f"  Total Entries: {daily_stats['total_entries']}")
        print(f"  Total Exits: {daily_stats['total_exits']}")
        
        if self.processing_times:
            avg_processing_time = np.mean(self.processing_times)
            print(f"\nPERFORMANCE:")
            print(f"  Average Processing Time: {avg_processing_time:.3f}s")
            print(f"  Estimated FPS: {1/avg_processing_time:.1f}")
        
        print("="*50 + "\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Real-time Intelligent Face Tracking System')
    parser.add_argument('--config', '-c', default='config.json', 
                       help='Path to configuration file')
    parser.add_argument('--video', '-v', help='Path to video file (overrides config)')
    parser.add_argument('--rtsp', '-r', help='RTSP URL (overrides config)')
    parser.add_argument('--no-display', action='store_true', 
                       help='Disable video display')
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = FaceTrackingSystem(args.config)
        
        # Override config with command line arguments
        if args.video:
            system.config['video_source'] = args.video
            system.config['use_rtsp'] = False
        
        if args.rtsp:
            system.config['rtsp_url'] = args.rtsp
            system.config['use_rtsp'] = True
        
        if args.no_display:
            system.config['display']['show_video'] = False
        
        # Run system
        system.run()
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
