#!/bin/bash
#
# Chatty AI Startup Script
# Plays intro video and launches the web interface
#

# Configuration
VIDEO_PATH="/home/nickspi5/Chatty_AI/Chatty_AI_starting.mp4"
CHATTY_URL="http://localhost:5000"
LOG_FILE="/home/nickspi5/Chatty_AI/logs/startup.log"
STARTUP_DELAY=75
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

# Wait for display to be ready
export DISPLAY=:0
export XAUTHORITY=/home/nickspi5/.Xauthority

# Check for development mode
DEV_MODE="${1:-production}"

# Ensure services are running
log_message "Starting Chatty AI services..."

# Start preloader if not running
if ! systemctl is-active --quiet chatty-ai-preloader.service; then
    log_message "Starting preloader service..."
    systemctl --user start chatty-ai-preloader.service 2>/dev/null || \
    sudo systemctl start chatty-ai-preloader.service
    sleep 5
fi

# Start main service if not running
if ! systemctl is-active --quiet chatty-ai.service; then
    log_message "Starting Chatty AI service..."
    systemctl --user start chatty-ai.service 2>/dev/null || \
    sudo systemctl start chatty-ai.service
    sleep 5
fi

# Check if video file exists and play it
if [ -f "$VIDEO_PATH" ]; then
    log_message "Playing startup video..."
    
    #  Play video once without retries
    cvlc --fullscreen --play-and-exit "$VIDEO_PATH" 2>/dev/null || log_message "Video playback ended"

    log_message "Video sequence complete"
else
    log_message "Warning: Video file not found at $VIDEO_PATH"
fi

# After video plays, wait for services to be ready
log_message "Waiting $STARTUP_DELAY seconds for services..."
sleep $STARTUP_DELAY

while [ $WAITED -lt $MAX_WAIT ]; do
    if netstat -tuln | grep -q ":5000 "; then
        log_message "Port 5000 is open and ready"
        sleep 5  # Wait to ensure fully ready
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    
    if [ $WAITED -eq 30 ] || [ $WAITED -eq 60 ]; then
        log_message "Still waiting for port 5000... ($WAITED seconds)"
    fi
done

# Kill VLC if still running
if [ -n "$VLC_PID" ] && ps -p $VLC_PID > /dev/null 2>&1; then
    kill $VLC_PID 2>/dev/null
fi

# Launch Chromium
if [ "$DEV_MODE" = "dev" ]; then
    log_message "Launching Chromium in development mode..."
    chromium-browser --start-maximized "$CHATTY_URL" 2>/dev/null &
else
    log_message "Launching Chromium in kiosk mode..."
    chromium-browser \
        --kiosk \
        --noerrdialogs \
        --disable-infobars \
        --start-maximized \
        --disable-translate \
        --disable-features=TranslateUI \
        --disable-component-update \
        "$CHATTY_URL" 2>/dev/null &
fi

CHROMIUM_PID=$!
log_message "Chromium launched with PID: $CHROMIUM_PID"

# Hide mouse cursor
if command -v unclutter &> /dev/null; then
    unclutter -idle 1 -root &
fi

log_message "Startup sequence complete!"
log_message "========================================="

# Keep script running
wait $CHROMIUM_PID
