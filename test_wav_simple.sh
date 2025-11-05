#!/bin/bash

echo "======================================================================"
echo "Testing MP3 with Python"
echo "======================================================================"
PROFILE_PERFORMANCE=1 pipenv run python openkeyscan_analyzer_server.py -w 1 <<EOF
{"id": "test-mp3", "path": "C:/Users/Chris/Music/Athys & Duster - Barfight.mp3"}
EOF

echo ""
echo "======================================================================"
echo "Testing WAV with Python"
echo "======================================================================"
PROFILE_PERFORMANCE=1 pipenv run python openkeyscan_analyzer_server.py -w 1 <<EOF
{"id": "test-wav", "path": "C:/Users/Chris/Music/audio formats/Andy C - Workout.wav"}
EOF
