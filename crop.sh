#!/usr/bin/env bash

# Check if the correct number of arguments is provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <input_mp3_path> <start_seconds> <end_seconds>"
    exit 1
fi

INPUT_FILE="$1"
START="$2"
END="$3"

# Extract the base name of the input file (without extension)
BASENAME=$(basename "$INPUT_FILE" .mp3)

# Construct the output file name
OUTPUT_FILE="${BASENAME}_${START}_${END}.mp3"

# Use ffmpeg to crop the MP3
ffmpeg -i "$INPUT_FILE" -ss "$START" -to "$END" -c copy "$OUTPUT_FILE"

# Check if ffmpeg was successful
if [ $? -eq 0 ]; then
    echo "Cropped file created: $OUTPUT_FILE"
else
    echo "An error occurred while creating the cropped file."
    exit 1
fi
