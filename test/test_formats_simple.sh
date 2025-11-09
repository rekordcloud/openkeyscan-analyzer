#!/bin/bash

# Test script for all audio formats
# Tests each format with profiling enabled

export PROFILE_PERFORMANCE=1

EXE="../dist/openkeyscan-analyzer/openkeyscan-analyzer"

# Test files
declare -A TEST_FILES=(
    ["MP3"]="/Users/chris/Music/cloud test/Mob Tactics - Now Is the Time.mp3"
    ["WAV"]="/Users/chris/Music/audio formats/Christian Smith - Burning Chrome.wav"
    ["FLAC"]="/Users/chris/Music/audio formats/Pleasurekraft - G.O.D. (Gospel of Doubt) ft Casey Gerald - Spektre Remix.flac"
    ["M4A"]="/Users/chris/Music/audio formats/Knife Party - Bonfire.m4a"
    ["OGG"]="/Users/chris/Music/audio formats/Burning Chrome - Christian Smith.ogg"
    ["AAC"]="/Users/chris/Music/audio formats/Burning Chrome - Christian Smith.aac"
    ["MP4"]="/Users/chris/Music/audio formats/serato.mp4"
)

echo "========================================================================"
echo "TESTING ALL AUDIO FORMATS - Executable (macOS)"
echo "========================================================================"
echo ""

for format in "${!TEST_FILES[@]}"; do
    file="${TEST_FILES[$format]}"

    echo "--------------------------------------------------------------------"
    echo "Testing $format: $(basename "$file")"
    echo "--------------------------------------------------------------------"

    if [ ! -f "$file" ]; then
        echo "❌ File not found: $file"
        continue
    fi

    # Start server and send request
    {
        echo "{\"id\": \"test\", \"path\": \"$file\"}"
        sleep 5
    } | "$EXE" -w 1 2>&1 &

    PID=$!

    # Wait for completion
    wait $PID

    echo ""
done

echo "========================================================================"
echo "Done!"
echo "========================================================================"
