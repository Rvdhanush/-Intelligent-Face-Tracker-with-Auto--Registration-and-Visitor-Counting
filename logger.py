"""
Logging module for face tracking system.
Handles file logging, image saving, and event logging.
"""

import os
import cv2
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path

class FaceTrackingLogger:
    def __init__(self, base_log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize the face tracking logger.
        
        Args:
            base_log_dir: Base directory for all logs
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.base_log_dir = Path(base_log_dir)
        self.log_level = getattr(logging, log_level.upper())
        
        # Create directory structure
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        # Event counters
        self.event_counters = {
            'entries': 0,
            'exits': 0,
            'new_faces': 0,
            'recognitions': 0
        }
        
        logging.info("Face tracking logger initialized")
    
    def setup_directories(self):
        """Create necessary directory structure."""
        # Main directories
        self.base_log_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        self.images_dir = self.base_log_dir / "images"
        self.entries_dir = self.images_dir / "entries"
        self.exits_dir = self.images_dir / "exits"
        self.faces_dir = self.images_dir / "faces"
        self.events_dir = self.base_log_dir / "events"
        
        # Create all directories
        for directory in [self.images_dir, self.entries_dir, self.exits_dir, 
                         self.faces_dir, self.events_dir]:
            directory.mkdir(exist_ok=True)
        
        # Create daily subdirectories
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_entries_dir = self.entries_dir / today
        self.daily_exits_dir = self.exits_dir / today
        self.daily_faces_dir = self.faces_dir / today
        
        for directory in [self.daily_entries_dir, self.daily_exits_dir, self.daily_faces_dir]:
            directory.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Setup file and console logging."""
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        log_file = self.base_log_dir / f"face_tracking_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(file_formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            handlers=[file_handler, console_handler]
        )
        
        # Create events log file
        self.events_log_file = self.events_dir / f"events_{datetime.now().strftime('%Y%m%d')}.log"
    
    def save_face_image(self, face_image: np.ndarray, face_id: str, 
                       event_type: str = "detection", 
                       bbox: Optional[Tuple[int, int, int, int]] = None,
                       confidence: Optional[float] = None) -> Optional[str]:
        """
        Save face image to appropriate directory.
        
        Args:
            face_image: Cropped face image
            face_id: Unique face identifier
            event_type: Type of event (entry, exit, detection, registration)
            bbox: Bounding box coordinates
            confidence: Detection confidence
            
        Returns:
            Path to saved image or None if failed
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
            
            # Determine save directory based on event type
            if event_type == "entry":
                save_dir = self.daily_entries_dir
            elif event_type == "exit":
                save_dir = self.daily_exits_dir
            else:
                save_dir = self.daily_faces_dir
            
            # Create filename
            filename = f"{face_id}_{event_type}_{timestamp}.jpg"
            file_path = save_dir / filename
            
            # Add metadata overlay if bbox and confidence provided
            if bbox and confidence:
                face_with_info = self._add_metadata_overlay(face_image, face_id, confidence, bbox)
            else:
                face_with_info = face_image.copy()
            
            # Save image
            success = cv2.imwrite(str(file_path), face_with_info)
            
            if success:
                logging.info(f"Saved {event_type} image for face {face_id}: {file_path}")
                return str(file_path)
            else:
                logging.error(f"Failed to save image for face {face_id}")
                return None
                
        except Exception as e:
            logging.error(f"Error saving face image for {face_id}: {e}")
            return None
    
    def _add_metadata_overlay(self, image: np.ndarray, face_id: str, 
                            confidence: float, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Add metadata overlay to face image."""
        img_with_overlay = image.copy()
        
        # Add text overlay
        text_lines = [
            f"ID: {face_id}",
            f"Conf: {confidence:.3f}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        y_offset = 20
        for line in text_lines:
            cv2.putText(img_with_overlay, line, (5, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
        
        return img_with_overlay
    
    def log_event(self, event_type: str, face_id: str, 
                  image_path: Optional[str] = None,
                  confidence: Optional[float] = None,
                  bbox: Optional[Tuple[int, int, int, int]] = None,
                  additional_data: Optional[Dict] = None) -> bool:
        """
        Log a face tracking event.
        
        Args:
            event_type: Type of event (entry, exit, registration, recognition)
            face_id: Face identifier
            image_path: Path to associated image
            confidence: Detection/recognition confidence
            bbox: Bounding box coordinates
            additional_data: Additional event data
            
        Returns:
            True if logged successfully
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Create event record
            event_record = {
                'timestamp': timestamp,
                'event_type': event_type,
                'face_id': face_id,
                'image_path': image_path,
                'confidence': confidence,
                'bbox': bbox,
                'additional_data': additional_data or {}
            }
            
            # Write to events log file
            with open(self.events_log_file, 'a') as f:
                f.write(json.dumps(event_record) + '\n')
            
            # Update counters
            if event_type == 'entry':
                self.event_counters['entries'] += 1
            elif event_type == 'exit':
                self.event_counters['exits'] += 1
            elif event_type == 'registration':
                self.event_counters['new_faces'] += 1
            elif event_type == 'recognition':
                self.event_counters['recognitions'] += 1
            
            # Log to main logger
            logging.info(f"Event logged: {event_type} for face {face_id}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error logging event {event_type} for face {face_id}: {e}")
            return False
    
    def log_face_entry(self, face_id: str, face_image: np.ndarray,
                      bbox: Tuple[int, int, int, int], confidence: float) -> bool:
        """Log a face entry event with image."""
        try:
            # Save entry image
            image_path = self.save_face_image(face_image, face_id, "entry", bbox, confidence)
            
            # Log event
            return self.log_event(
                event_type="entry",
                face_id=face_id,
                image_path=image_path,
                confidence=confidence,
                bbox=bbox
            )
            
        except Exception as e:
            logging.error(f"Error logging face entry for {face_id}: {e}")
            return False
    
    def log_face_exit(self, face_id: str, face_image: Optional[np.ndarray] = None,
                     bbox: Optional[Tuple[int, int, int, int]] = None, 
                     confidence: Optional[float] = None) -> bool:
        """Log a face exit event with optional image."""
        try:
            image_path = None
            if face_image is not None:
                image_path = self.save_face_image(face_image, face_id, "exit", bbox, confidence)
            
            # Log event
            return self.log_event(
                event_type="exit",
                face_id=face_id,
                image_path=image_path,
                confidence=confidence,
                bbox=bbox
            )
            
        except Exception as e:
            logging.error(f"Error logging face exit for {face_id}: {e}")
            return False
    
    def log_face_registration(self, face_id: str, face_image: np.ndarray,
                            embedding: np.ndarray, confidence: float) -> bool:
        """Log a new face registration."""
        try:
            # Save registration image
            image_path = self.save_face_image(face_image, face_id, "registration")
            
            # Log event with embedding info
            additional_data = {
                'embedding_shape': embedding.shape,
                'embedding_norm': float(np.linalg.norm(embedding))
            }
            
            return self.log_event(
                event_type="registration",
                face_id=face_id,
                image_path=image_path,
                confidence=confidence,
                additional_data=additional_data
            )
            
        except Exception as e:
            logging.error(f"Error logging face registration for {face_id}: {e}")
            return False
    
    def log_face_recognition(self, face_id: str, confidence: float,
                           bbox: Tuple[int, int, int, int]) -> bool:
        """Log a face recognition event."""
        try:
            return self.log_event(
                event_type="recognition",
                face_id=face_id,
                confidence=confidence,
                bbox=bbox
            )
            
        except Exception as e:
            logging.error(f"Error logging face recognition for {face_id}: {e}")
            return False
    
    def log_system_event(self, event_type: str, message: str, 
                        additional_data: Optional[Dict] = None):
        """Log system-level events."""
        try:
            timestamp = datetime.now().isoformat()
            
            system_event = {
                'timestamp': timestamp,
                'event_type': f"system_{event_type}",
                'message': message,
                'additional_data': additional_data or {}
            }
            
            with open(self.events_log_file, 'a') as f:
                f.write(json.dumps(system_event) + '\n')
            
            logging.info(f"System event: {event_type} - {message}")
            
        except Exception as e:
            logging.error(f"Error logging system event: {e}")
    
    def get_daily_summary(self, date: Optional[str] = None) -> Dict:
        """Get summary of daily events."""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            events_file = self.events_dir / f"events_{date.replace('-', '')}.log"
            
            if not events_file.exists():
                return {
                    'date': date,
                    'entries': 0,
                    'exits': 0,
                    'registrations': 0,
                    'recognitions': 0,
                    'unique_faces': set()
                }
            
            summary = {
                'date': date,
                'entries': 0,
                'exits': 0,
                'registrations': 0,
                'recognitions': 0,
                'unique_faces': set()
            }
            
            with open(events_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        event_type = event.get('event_type', '')
                        face_id = event.get('face_id', '')
                        
                        if event_type == 'entry':
                            summary['entries'] += 1
                        elif event_type == 'exit':
                            summary['exits'] += 1
                        elif event_type == 'registration':
                            summary['registrations'] += 1
                        elif event_type == 'recognition':
                            summary['recognitions'] += 1
                        
                        if face_id:
                            summary['unique_faces'].add(face_id)
                            
                    except json.JSONDecodeError:
                        continue
            
            # Convert set to count
            summary['unique_faces'] = len(summary['unique_faces'])
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating daily summary: {e}")
            return {'date': date, 'entries': 0, 'exits': 0, 'registrations': 0, 
                   'recognitions': 0, 'unique_faces': 0}
    
    def get_current_stats(self) -> Dict:
        """Get current session statistics."""
        return {
            'session_stats': self.event_counters.copy(),
            'daily_summary': self.get_daily_summary(),
            'log_directories': {
                'base': str(self.base_log_dir),
                'images': str(self.images_dir),
                'events': str(self.events_dir)
            }
        }
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files and images."""
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            
            # Clean up old directories
            for directory in [self.entries_dir, self.exits_dir, self.faces_dir]:
                for subdir in directory.iterdir():
                    if subdir.is_dir() and subdir.stat().st_mtime < cutoff_date:
                        import shutil
                        shutil.rmtree(subdir)
                        logging.info(f"Cleaned up old directory: {subdir}")
            
            # Clean up old event logs
            for log_file in self.events_dir.glob("events_*.log"):
                if log_file.stat().st_mtime < cutoff_date:
                    log_file.unlink()
                    logging.info(f"Cleaned up old log file: {log_file}")
            
        except Exception as e:
            logging.error(f"Error cleaning up old logs: {e}")
