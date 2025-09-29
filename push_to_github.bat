@echo off
echo 🚀 Pushing Face Tracking System to GitHub
echo ============================================

echo 📁 Navigating to project directory...
cd /d "c:\Desktop\katomaran"

echo 🔧 Initializing Git repository...
git init

echo 🔗 Adding remote repository...
git remote add origin https://github.com/Rvdhanush/-Intelligent-Face-Tracker-with-Auto--Registration-and-Visitor-Counting.git

echo 📦 Adding all files to staging...
git add .

echo 📝 Creating commit...
git commit -m "🏆 Complete Face Tracking System - Hackathon Submission

✅ Mandatory Tech Stack Implemented:
- YOLOv8 Face Detection (detector.py)
- InsightFace/ArcFace Recognition (recognizer.py, onnx_recognizer.py)
- ByteTrack Multi-Object Tracking (bytetrack.py)
- SQLite Database with comprehensive schema (db.py)
- JSON Configuration system (config.json)
- Complete logging: files + images + database (logger.py)

🚀 Key Features:
- Real-time face detection and recognition
- Entry/exit event detection and logging
- Web dashboard with live video feed (app.py)
- Multiple deployment options (main.py, production_main.py)
- Clean, modular Python architecture
- Professional documentation (README.md)

🎯 Hackathon Compliance:
- All mandatory requirements met
- Clean directory structure (18 essential files)
- Production-ready code quality
- Comprehensive setup instructions

This project is part of a hackathon run by https://katomaran.com"

echo 🚀 Pushing to GitHub...
git push -u origin main

echo ✅ Push completed! Check your repository at:
echo https://github.com/Rvdhanush/-Intelligent-Face-Tracker-with-Auto--Registration-and-Visitor-Counting

pause
