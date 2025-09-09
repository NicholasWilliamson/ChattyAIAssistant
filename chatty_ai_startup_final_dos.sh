#!/bin/bash
#
# Chatty AI Startup Script
# Shows black screen, video, loading animation, then main interface
#

# Configuration
VIDEO_PATH="/home/nickspi5/Chatty_AI/Chatty_AI_starting.mp4"
LOADING_PAGE="/home/nickspi5/Chatty_AI/templates/chatty_loading.html"
BLACK_IMAGE="/home/nickspi5/Chatty_AI/templates/black.png"
CHATTY_URL="http://localhost:5000"
LOG_FILE="/home/nickspi5/Chatty_AI/logs/startup.log"
USER="nickspi5"

# Create log directory if it doesn't exist
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

# Wait for desktop to be fully loaded
log_message "Waiting for desktop to load..."
sleep 15

# Kill any error dialogs
wmctrl -c "Desktop manager is not active" 2>/dev/null

# Display black screen in fullscreen browser window (covers desktop)
log_message "Displaying black screen..."
chromium-browser \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --start-fullscreen \
    --disable-session-crashed-bubble \
    --check-for-update-interval=31536000 \
    "file://$BLACK_IMAGE" 2>/dev/null &

BLACK_PID=$!
sleep 2

# Play video
if [ -f "$VIDEO_PATH" ]; then
    log_message "Playing startup video..."
    
    # Kill the black screen browser
    kill $BLACK_PID 2>/dev/null
    
    # Play video in fullscreen
    cvlc --fullscreen --play-and-exit "$VIDEO_PATH" 2>/dev/null || log_message "Video playback ended"
    
    log_message "Video sequence complete"
else
    log_message "Warning: Video file not found at $VIDEO_PATH"
    # Kill black screen if video doesn't exist
    kill $BLACK_PID 2>/dev/null
fi

# Launch loading screen
log_message "Launching loading screen..."

# Clear any leftover browser instances
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
log_message "System loading sequence in progress..."
log_message "========================================="

# Keep script running
wait $CHROMIUM_PID