"""
Database module for face tracking system.
Handles SQLite operations for storing face data, events, and visitor counts.
"""

import sqlite3
import numpy as np
import json
from datetime import datetime
from typing import Optional, List, Tuple, Dict
import logging

class FaceDatabase:
    def __init__(self, db_path: str = "face_tracking.db"):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create faces table to store registered faces
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT UNIQUE NOT NULL,
                    embedding BLOB NOT NULL,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_visits INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Create events table to log all face entry/exit events
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK (event_type IN ('entry', 'exit', 'registration', 'recognition')),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT,
                    confidence REAL,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_width INTEGER,
                    bbox_height INTEGER,
                    FOREIGN KEY (face_id) REFERENCES faces (face_id)
                )
            ''')
            
            # Create visitor_stats table for quick analytics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS visitor_stats (
                    id INTEGER PRIMARY KEY,
                    date TEXT UNIQUE NOT NULL,
                    unique_visitors INTEGER DEFAULT 0,
                    total_entries INTEGER DEFAULT 0,
                    total_exits INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logging.info("Database initialized successfully")
    
    def register_face(self, face_id: str, embedding: np.ndarray) -> bool:
        """Register a new face with its embedding."""
        try:
            embedding_blob = embedding.tobytes()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO faces (face_id, embedding, first_seen, last_seen)
                    VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ''', (face_id, embedding_blob))
                conn.commit()
                logging.info(f"Registered new face: {face_id}")
                return True
        except sqlite3.IntegrityError:
            logging.warning(f"Face {face_id} already exists in database")
            return False
        except Exception as e:
            logging.error(f"Error registering face {face_id}: {e}")
            return False
    
    def get_face_embedding(self, face_id: str) -> Optional[np.ndarray]:
        """Retrieve face embedding by face_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT embedding FROM faces WHERE face_id = ?', (face_id,))
                result = cursor.fetchone()
                if result:
                    return np.frombuffer(result[0], dtype=np.float32)
                return None
        except Exception as e:
            logging.error(f"Error retrieving embedding for {face_id}: {e}")
            return None
    
    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Get all face embeddings for recognition."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT face_id, embedding FROM faces WHERE is_active = 1')
                results = cursor.fetchall()
                embeddings = []
                for face_id, embedding_blob in results:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    embeddings.append((face_id, embedding))
                return embeddings
        except Exception as e:
            logging.error(f"Error retrieving all embeddings: {e}")
            return []
    
    def update_face_last_seen(self, face_id: str):
        """Update the last seen timestamp for a face."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE faces 
                    SET last_seen = CURRENT_TIMESTAMP, total_visits = total_visits + 1
                    WHERE face_id = ?
                ''', (face_id,))
                conn.commit()
        except Exception as e:
            logging.error(f"Error updating last seen for {face_id}: {e}")
    
    def log_event(self, event_type: str, face_id: str, image_path: str = None, 
                  confidence: float = None, bbox: Tuple[int, int, int, int] = None) -> bool:
        """Log a face entry or exit event."""
        try:
            bbox_x, bbox_y, bbox_w, bbox_h = bbox if bbox else (None, None, None, None)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO events (face_id, event_type, image_path, confidence, 
                                      bbox_x, bbox_y, bbox_width, bbox_height)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (face_id, event_type, image_path, confidence, bbox_x, bbox_y, bbox_w, bbox_h))
                conn.commit()
                logging.info(f"Logged {event_type} event for face {face_id}")
                
                # Update daily stats
                self._update_daily_stats(event_type)
                return True
        except Exception as e:
            logging.error(f"Error logging event for {face_id}: {e}")
            return False
    
    def _update_daily_stats(self, event_type: str):
        """Update daily visitor statistics."""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if today's record exists
                cursor.execute('SELECT id FROM visitor_stats WHERE date = ?', (today,))
                if cursor.fetchone():
                    # Update existing record
                    if event_type == 'entry':
                        cursor.execute('''
                            UPDATE visitor_stats 
                            SET total_entries = total_entries + 1, last_updated = CURRENT_TIMESTAMP
                            WHERE date = ?
                        ''', (today,))
                    elif event_type == 'exit':
                        cursor.execute('''
                            UPDATE visitor_stats 
                            SET total_exits = total_exits + 1, last_updated = CURRENT_TIMESTAMP
                            WHERE date = ?
                        ''', (today,))
                else:
                    # Create new record
                    entries = 1 if event_type == 'entry' else 0
                    exits = 1 if event_type == 'exit' else 0
                    cursor.execute('''
                        INSERT INTO visitor_stats (date, total_entries, total_exits)
                        VALUES (?, ?, ?)
                    ''', (today, entries, exits))
                
                conn.commit()
        except Exception as e:
            logging.error(f"Error updating daily stats: {e}")
    
    def get_unique_visitor_count(self, date: str = None) -> int:
        """Get unique visitor count for a specific date or all time."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if date:
                    cursor.execute('''
                        SELECT COUNT(DISTINCT face_id) FROM events 
                        WHERE DATE(timestamp) = ? AND event_type = 'entry'
                    ''', (date,))
                else:
                    cursor.execute('SELECT COUNT(*) FROM faces WHERE is_active = 1')
                
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logging.error(f"Error getting unique visitor count: {e}")
            return 0
    
    def get_daily_stats(self, date: str = None) -> Dict:
        """Get daily statistics for a specific date or today."""
        try:
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT unique_visitors, total_entries, total_exits, last_updated
                    FROM visitor_stats WHERE date = ?
                ''', (date,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        'date': date,
                        'unique_visitors': result[0],
                        'total_entries': result[1],
                        'total_exits': result[2],
                        'last_updated': result[3]
                    }
                else:
                    # Calculate from events table if no stats record exists
                    unique_count = self.get_unique_visitor_count(date)
                    cursor.execute('''
                        SELECT COUNT(*) FROM events 
                        WHERE DATE(timestamp) = ? AND event_type = 'entry'
                    ''', (date,))
                    entries = cursor.fetchone()[0]
                    
                    cursor.execute('''
                        SELECT COUNT(*) FROM events 
                        WHERE DATE(timestamp) = ? AND event_type = 'exit'
                    ''', (date,))
                    exits = cursor.fetchone()[0]
                    
                    return {
                        'date': date,
                        'unique_visitors': unique_count,
                        'total_entries': entries,
                        'total_exits': exits,
                        'last_updated': datetime.now().isoformat()
                    }
        except Exception as e:
            logging.error(f"Error getting daily stats: {e}")
            return {'date': date, 'unique_visitors': 0, 'total_entries': 0, 'total_exits': 0}
    
    def get_recent_events(self, limit: int = 10) -> List[Dict]:
        """Get recent face events."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT face_id, event_type, timestamp, image_path, confidence
                    FROM events 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                results = cursor.fetchall()
                events = []
                for row in results:
                    events.append({
                        'face_id': row[0],
                        'event_type': row[1],
                        'timestamp': row[2],
                        'image_path': row[3],
                        'confidence': row[4]
                    })
                return events
        except Exception as e:
            logging.error(f"Error getting recent events: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data beyond specified days."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM events 
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days_to_keep))
                
                cursor.execute('''
                    DELETE FROM visitor_stats 
                    WHERE date < date('now', '-{} days')
                '''.format(days_to_keep))
                
                conn.commit()
                logging.info(f"Cleaned up data older than {days_to_keep} days")
        except Exception as e:
            logging.error(f"Error cleaning up old data: {e}")
    
    def close(self):
        """Close database connection."""
        pass  # Using context managers, so no explicit close needed
