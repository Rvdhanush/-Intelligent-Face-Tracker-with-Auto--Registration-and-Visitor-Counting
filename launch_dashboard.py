"""
Launch script for the Face Tracking System Web Dashboard.
"""

import os
import sys
import webbrowser
import time
import threading
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'flask', 'flask_socketio', 'cv2', 'numpy', 
        'sqlite3', 'ultralytics', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'flask_socketio':
                import flask_socketio
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages and try again.")
        return False
    
    return True

def check_system_files():
    """Check if all required system files exist."""
    required_files = [
        'app.py',
        'templates/dashboard.html',
        'db.py',
        'detector.py',
        'simple_main.py',
        'config.json'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def open_browser():
    """Open browser after a short delay."""
    time.sleep(3)  # Wait for server to start
    webbrowser.open('http://localhost:5000')

def main():
    """Main launcher function."""
    print("🚀 Face Tracking System - Web Dashboard Launcher")
    print("="*60)
    
    # Check dependencies
    print("📋 Checking dependencies...")
    if not check_dependencies():
        return
    print("✅ All dependencies found")
    
    # Check system files
    print("📁 Checking system files...")
    if not check_system_files():
        return
    print("✅ All system files found")
    
    # Check video files
    video_dir = Path("Video Datasets")
    if video_dir.exists():
        video_count = len(list(video_dir.glob("*.mp4")))
        print(f"🎥 Found {video_count} video files for processing")
    else:
        print("⚠️  No 'Video Datasets' directory found")
    
    # Check database
    if Path("face_tracking.db").exists():
        print("🗄️  Database file found")
    else:
        print("🗄️  Database will be created on first run")
    
    print("\n🌐 Starting Web Dashboard...")
    print("📊 Dashboard URL: http://localhost:5000")
    print("🎯 Features: Real-time monitoring, Live video feed, Statistics")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*60)
    
    # Open browser in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    try:
        from app import app, socketio
        socketio.run(app, debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    main()
