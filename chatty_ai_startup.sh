#!/bin/bash
#
# Chatty AI Startup Script
# Plays intro video and launches the web interface
#

# Configuration
VIDEO_PATH="/home/nickspi5/Chatty_AI/Chatty_AI_starting.mp4"
CHATTY_URL="http://localhost:5000"
LOG_FILE="/home/nickspi5/Chatty_AI/logs/startup.log"
STARTUP_DELAY=70
USER="nickspi5"

# Create log directory if it doesn't exist
mkdir -p /home/nickspi5/Chatty_AI/logs

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if service is running
check_service() {
    systemctl is-active --quiet "$1"
    return $?
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local max_wait=$2
    local count=0
    
    log_message "Waiting for $service to start..."
    
    while [ $count -lt $max_wait ]; do
        if check_service "$service"; then
            log_message "$service is running"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done
    
    log_message "Warning: $service did not start within $max_wait seconds"
    return 1
}

# Function to check if port is open
check_port() {
    netstat -tuln | grep -q ":$1 "
    return $?
}

# Function to wait for port
wait_for_port() {
    local port=$1
    local max_wait=$2
    local count=0
    
    log_message "Waiting for port $port to open..."
    
    while [ $count -lt $max_wait ]; do
        if check_port "$port"; then
            log_message "Port $port is open"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done
    
    log_message "Warning: Port $port did not open within $max_wait seconds"
    return 1
}

# Start logging
log_message "========================================="
log_message "Chatty AI Startup Sequence Initiated"
log_message "========================================="

# Wait for display to be ready (if running at boot)
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi

# Wait for X server to be ready
sleep 5

# Check if video file exists
if [ ! -f "$VIDEO_PATH" ]; then
    log_message "Warning: Video file not found at $VIDEO_PATH"
    log_message "Skipping video playback"
else
    log_message "Playing startup video..."
    
    # Kill any existing VLC instances
    pkill -f vlc 2>/dev/null
    
    # Play video in fullscreen with audio
    # Using cvlc (command line VLC) for better control
    sudo -u "$USER" DISPLAY=:0 cvlc \
        --fullscreen \
        --no-video-title-show \
        --play-and-exit \
        --no-loop \
        --no-repeat \
        --intf dummy \
        "$VIDEO_PATH" 2>/dev/null &
    
    VLC_PID=$!
    log_message "VLC started with PID: $VLC_PID"
fi

# Start checking services while video plays
log_message "Checking system services..."

# Check preloader service
if ! check_service "chatty-ai-preloader.service"; then
    log_message "Starting preloader service..."
    sudo systemctl start chatty-ai-preloader.service
fi
wait_for_service "chatty-ai-preloader.service" 30

# Check main Chatty AI service
if ! check_service "chatty-ai.service"; then
    log_message "Starting Chatty AI service..."
    sudo systemctl start chatty-ai.service
fi
wait_for_service "chatty-ai.service" 30

# Wait for web server to be ready
wait_for_port 5000 60

# Calculate remaining wait time
if [ -n "$VLC_PID" ]; then
    log_message "Waiting for startup sequence to complete..."
    
    # Wait for the specified delay or until video finishes
    WAITED=0
    while [ $WAITED -lt $STARTUP_DELAY ]; do
        # Check if VLC is still running
        if ! ps -p $VLC_PID > /dev/null 2>&1; then
            log_message "Video playback completed"
            # Wait remaining time
            REMAINING=$((STARTUP_DELAY - WAITED))
            if [ $REMAINING -gt 0 ]; then
                log_message "Waiting additional $REMAINING seconds for services to stabilize..."
                sleep $REMAINING
            fi
            break
        fi
        sleep 1
        WAITED=$((WAITED + 1))
    done
    
    # Kill VLC if still running
    if ps -p $VLC_PID > /dev/null 2>&1; then
        log_message "Stopping video playback..."
        kill $VLC_PID 2>/dev/null
        sleep 1
    fi
else
    # No video, just wait for services
    log_message "Waiting $STARTUP_DELAY seconds for services to stabilize..."
    sleep $STARTUP_DELAY
fi

# Kill any existing Chromium instances
pkill -f chromium 2>/dev/null
sleep 2

# Launch Chromium in kiosk mode
log_message "Launching Chromium in kiosk mode..."

# Clear Chromium crash flag
sed -i 's/"exited_cleanly":false/"exited_cleanly":true/' /home/$USER/.config/chromium/Default/Preferences 2>/dev/null
sed -i 's/"exit_type":"Crashed"/"exit_type":"Normal"/' /home/$USER/.config/chromium/Default/Preferences 2>/dev/null

# Launch Chromium with optimal settings for Raspberry Pi
sudo -u "$USER" DISPLAY=:0 chromium-browser \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --disable-features=TranslateUI \
    --disable-web-security \
    --disable-features=InfiniteSessionRestore \
    --check-for-update-interval=31536000 \
    --disable-component-update \
    --autoplay-policy=no-user-gesture-required \
    --start-maximized \
    --window-position=0,0 \
    --disable-pinch \
    --overscroll-history-navigation=0 \
    "$CHATTY_URL" 2>/dev/null &

CHROMIUM_PID=$!
log_message "Chromium launched with PID: $CHROMIUM_PID"

# Optional: Hide mouse cursor after a few seconds
sleep 5
if command -v unclutter &> /dev/null; then
    log_message "Hiding mouse cursor..."
    sudo -u "$USER" DISPLAY=:0 unclutter -idle 1 -root &
fi

log_message "========================================="
log_message "Chatty AI Startup Sequence Complete!"
log_message "System should now be fully operational"
log_message "========================================="

# Keep script running to maintain process
wait $CHROMIUM_PID