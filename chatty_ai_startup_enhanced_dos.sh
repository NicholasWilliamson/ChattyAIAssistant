#!/bin/bash
#
# Enhanced Chatty AI Startup Script with Loading Screen
# Black screen -> Video -> Loading animation -> Main interface
#

# Configuration
VIDEO_PATH="/home/nickspi5/Chatty_AI/Chatty_AI_starting.mp4"
LOADING_PAGE="/home/nickspi5/Chatty_AI/templates/chatty_loading.html"
CHATTY_URL="http://localhost:5000"
LOG_FILE="/home/nickspi5/Chatty_AI/logs/startup.log"
USER="nickspi5"

# Create log directory if it doesn't exist
mkdir -p /home/nickspi5/Chatty_AI/logs

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to hide desktop and show black screen
hide_desktop() {
    # Kill desktop panels and file manager
    pkill lxpanel 2>/dev/null
    pkill pcmanfm 2>/dev/null
    
    # Set black wallpaper
    pcmanfm --set-wallpaper=/usr/share/pixmaps/black.png 2>/dev/null || \
    feh --bg-fill /usr/share/pixmaps/black.png 2>/dev/null || \
    xsetroot -solid black 2>/dev/null
    
    # Hide mouse cursor
    unclutter -idle 0 &
}

# Start logging
log_message "========================================="
log_message "Enhanced Chatty AI Startup Sequence"
log_message "========================================="

# Set display
export DISPLAY=:0
export XAUTHORITY=/home/nickspi5/.Xauthority

# Wait for X to be ready
sleep 5

# Hide desktop and show black screen
log_message "Hiding desktop..."
hide_desktop

# Wait a moment for black screen
sleep 2

# Check if video file exists and play it
if [ -f "$VIDEO_PATH" ]; then
    log_message "Playing startup video..."
    
    # Play video in fullscreen
    cvlc --fullscreen --play-and-exit "$VIDEO_PATH" 2>/dev/null || log_message "Video playback ended"
    
    log_message "Video sequence complete"
else
    log_message "Warning: Video file not found at $VIDEO_PATH"
fi

# Launch loading screen in Chromium
log_message "Launching loading screen..."

# Kill any existing Chromium
pkill -f chromium 2>/dev/null
sleep 1

# Launch loading screen in fullscreen
chromium-browser \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --start-fullscreen \
    --disable-translate \
    --disable-features=TranslateUI \
    --disable-component-update \
    --disable-session-crashed-bubble \
    --check-for-update-interval=31536000 \
    "file://$LOADING_PAGE" 2>/dev/null &

CHROMIUM_PID=$!
log_message "Loading screen launched with PID: $CHROMIUM_PID"

# The loading page will automatically redirect to localhost:5000 after 70 seconds
# Just keep the script running
log_message "System loading sequence in progress..."

# Keep script running
wait $CHROMIUM_PID