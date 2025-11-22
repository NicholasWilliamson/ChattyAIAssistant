#!/bin/bash
#
# Chatty AI Startup Script
#

# Configuration
VIDEO_PATH="/home/nickspi5/Chatty_AI/Chatty_AI_starting.mp4"
LOADING_PAGE="/home/nickspi5/Chatty_AI/templates/chatty_loading.html"
BLACK_PAGE="/home/nickspi5/Chatty_AI/templates/black.html"
CHATTY_URL="http://localhost:5000"
LOG_FILE="/home/nickspi5/Chatty_AI/logs/startup.log"

# Create log directory
mkdir -p /home/nickspi5/Chatty_AI/logs

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start logging
log_message "========================================="
log_message "Chatty AI Startup Sequence Initiated"
log_message "========================================="

# Set display
export DISPLAY=:0
export XAUTHORITY=/home/nickspi5/.Xauthority

# Wait for X server to be ready
log_message "Waiting for X server..."
while ! xset q &>/dev/null; do
    sleep 0.5
done

# Hide desktop elements
log_message "Hiding desktop elements..."
pkill lxpanel 2>/dev/null
pkill pcmanfm 2>/dev/null
sleep 0.2

# STEP 1: Show black screen
log_message "Displaying black screen..."
chromium-browser \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --disable-restore-session-state \
    --disable-web-security \
    --start-fullscreen \
    --window-position=0,0 \
    --window-size=1920,1080 \
    "file://$BLACK_PAGE" 2>/dev/null &

BLACK_PID=$!
sleep 2  # Wait for black screen to render

# Ensure desktop is completely hidden
pcmanfm --desktop-off 2>/dev/null

# STEP 2: CLOSE black screen, THEN play video (so video replaces black screen)
log_message "Closing black screen before video..."
kill $BLACK_PID 2>/dev/null
sleep 0.5

if [ -f "$VIDEO_PATH" ]; then
    log_message "Playing startup video..."
    
    # Video is 15.02 seconds, so we know exact duration
    log_message "Video duration: 15 seconds"
    
    # Play video with optimal settings
    cvlc \
        --fullscreen \
        --no-video-title-show \
        --no-osd \
        --quiet \
        --intf dummy \
        --play-and-exit \
        "$VIDEO_PATH" 2>/dev/null &
    
    VLC_PID=$!
    
    # Wait for exact video duration (15 seconds + 1 second buffer)
    sleep 16
    
    # Force kill VLC if still running
    if ps -p $VLC_PID > /dev/null; then
        kill $VLC_PID 2>/dev/null
        sleep 1
    fi
    
    log_message "Video playback complete"
else
    log_message "Video not found at $VIDEO_PATH"
    sleep 15  # Show black screen for 15 seconds if no video
fi

# Kill any remaining VLC processes
pkill vlc 2>/dev/null
pkill cvlc 2>/dev/null
sleep 1

# STEP 3: Show loading screen for FULL 74 seconds
log_message "Launching loading screen for 74 seconds ..."
chromium-browser \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --disable-restore-session-state \
    --disable-web-security \
    --start-fullscreen \
    "file://$LOADING_PAGE" 2>/dev/null &

LOADING_PID=$!

# Wait for FULL loading screen duration (74 seconds)
log_message "Loading screen running for 74 seconds ..."
sleep 74

# Kill loading screen
log_message "Loading screen complete, closing..."
kill $LOADING_PID 2>/dev/null
sleep 0.5

# STEP 4: Launch main Chatty AI interface
log_message "Launching Chatty AI interface..."
chromium-browser \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --disable-restore-session-state \
    --disable-web-security \
    --start-fullscreen \
    "$CHATTY_URL" 2>/dev/null &

MAIN_PID=$!
log_message "Chatty AI interface launched successfully"
log_message "========================================="

# Keep the main interface running
wait $MAIN_PID
