#!/bin/bash

# Wait for system to fully load
sleep 5

# Launch Chromium in kiosk mode with autoplay and audio enabled
chromium-browser \
  --kiosk \
  --autoplay-policy=no-user-gesture-required \
  --no-sandbox \
  --disable-gpu \
  --disable-software-rasterizer \
  --enable-features=AutoplayIgnoreWebAudio \
  --use-gl=egl \
  --start-fullscreen \
  --start-maximized \
  --disable-translate \
  --no-first-run \
  --no-default-browser-check \
  --disable-infobars \
  --use-fake-ui-for-media-stream \
  --enable-features=MediaSessionService \
  --enable-features=HardwareMediaKeyHandling \  
  file:///home/nickspi5/Chatty_AI_UI/index.html
