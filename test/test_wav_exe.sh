#!/bin/bash

echo "======================================================================"
echo "Testing WAV with Executable"
echo "======================================================================"
PROFILE_PERFORMANCE=1 ../dist/openkeyscan-analyzer/openkeyscan-analyzer.exe -w 1 <<EOF
{"id": "test-wav", "path": "C:/Users/Chris/Music/audio formats/Andy C - Workout.wav"}
EOF
