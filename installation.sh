#!/bin/bash

# Enhanced Chatty AI Assistant Installation Script
# Run this script from your /home/nickspi5/Chatty_AI directory

echo "=== Enhanced Chatty AI Assistant Installation ==="

# Activate virtual environment
echo "Activating chatty-venv virtual environment..."
source chatty-venv/bin/activate

# Install additional Python packages
echo "Installing additional Python packages..."
pip install --upgrade pip

# Core packages for wake word detection and audio
pip install pvporcupine pygame

# Audio processing and threading utilities
pip install pyaudio webrtcvad

# Face Recognition utilities
pip install face-recognition opencv-python imutils

# Ensure existing packages are up to date
pip install --upgrade sounddevice soundfile
pip install --upgrade faster-whisper llama-cpp-python
pip install --upgrade picamera2

# Create necessary directories
echo "Creating directory structure..."
mkdir -p /home/nickspi5/Chatty_AI/dataset
mkdir -p /home/nickspi5/Chatty_AI/security_photos
mkdir -p /home/nickspi5/Chatty_AI/audio_files

# Download a beep sound file for wake word acknowledgment
echo "Downloading beep sound..."
wget -O /home/nickspi5/Chatty_AI/audio_files/beep.wav "https://www.soundjay.com/misc/sounds/beep-28.wav" || echo "Manual beep file creation needed"

# Create a simple beep sound if download fails
if [ ! -f "/home/nickspi5/Chatty_AI/audio_files/beep.wav" ]; then
    echo "Creating simple beep sound with ffmpeg (if available)..."
    which ffmpeg && ffmpeg -f lavfi -i "sine=frequency=1000:duration=0.5" /home/nickspi5/Chatty_AI/audio_files/beep.wav
fi

# Set proper permissions
echo "Setting permissions..."
chmod +x /home/nickspi5/Chatty_AI/*.py
chmod -R 755 /home/nickspi5/Chatty_AI/

# Create sample greeting responses file
echo "Creating greeting responses..."
cat > /home/nickspi5/Chatty_AI/greeting_responses.txt << 'EOF'
How are you doing today?
It's wonderful to see you again!
What brings you here today?
I hope you're having a great day!
Nice to see you! What can I help you with?
Good to have you back!
What's on your mind today?
How can I assist you?
It's always a pleasure to see you!
What would you like to talk about?
I'm here and ready to help!
How has your day been so far?
What can I do for you today?
Great to see you again!
I'm excited to chat with you!
What's new with you today?
How can I be of service?
Welcome back! What's happening?
It's lovely to see your face!
What adventures await us today?
EOF

# Create waiting responses file
cat > /home/nickspi5/Chatty_AI/waiting_responses.txt << 'EOF'
I'm still here if you need anything
Just checking in to see if you need help
I'm ready whenever you are
Still here and waiting to assist
Is there anything I can help you with?
I'm here if you want to chat
Just letting you know I'm available
Ready to help whenever you need me
I'm here and listening
Still standing by if you need assistance
Waiting patiently to be of service
I'm here if you'd like to talk
Ready and waiting for your questions
Just checking if there's anything you need
I remain at your service
Still here, still ready to help
Monitoring and ready to assist
Available whenever you're ready
I'm here, just say the word
Standing by for any questions
EOF

# Create jokes file
cat > /home/nickspi5/Chatty_AI/jokes.txt << 'EOF'
Why don't scientists trust atoms? Because they make up everything!
I told my wife she was drawing her eyebrows too high. She looked surprised.
Why don't eggs tell jokes? They'd crack each other up!
What do you call a fake noodle? An impasta!
Why did the scarecrow win an award? He was outstanding in his field!
What do you call a bear with no teeth? A gummy bear!
Why don't skeletons fight each other? They don't have the guts!
What's orange and sounds like a parrot? A carrot!
Why did the math book look so sad? Because it was full of problems!
What do you call a sleeping bull? A bulldozer!
Why don't oysters donate? Because they are shellfish!
What do you call a dinosaur that crashes his car? Tyrannosaurus Wrecks!
Why did the cookie go to the doctor? Because it felt crumbly!
What do you call a fish wearing a bowtie? Sofishticated!
Why don't scientists trust stairs? Because they're always up to something!
What did one wall say to the other wall? I'll meet you at the corner!
Why did the bicycle fall over? Because it was two tired!
What do you call a cow with no legs? Ground beef!
Why did the banana go to the doctor? It wasn't peeling well!
What do you call a pig that does karate? A pork chop!
EOF

# Create listening responses file
cat > /home/nickspi5/Chatty_AI/listening_responses.txt << 'EOF'
I'm listening, what would you like to know?
Yes, I'm here! What can I help you with?
You have my full attention, go ahead!
I'm ready to help, what's your question?
Yes, I heard you! What do you need?
I'm all ears, please continue!
You called and I'm here to assist!
Ready to listen, what's on your mind?
I'm focused on you, what can I do?
Yes, I'm paying attention! Go ahead!
You've got my attention, what's up?
I'm here and ready to help you out!
Listening carefully, what do you need?
You summoned me and I'm here to serve!
I'm tuned in, what would you like to discuss?
Ready and waiting for your question!
I heard the wake word, I'm listening!
You have my undivided attention!
I'm present and ready to assist you!
Yes, I'm actively listening to you!
EOF

# Create warning responses file
cat > /home/nickspi5/Chatty_AI/warning_responses.txt << 'EOF'
Attention: Unregistered person detected. Please identify yourself.
Warning: Unknown individual in the area. Security photo captured.
Alert: Unrecognized person detected. Photo taken for security purposes.
Notice: Unknown person detected. Please state your business.
Security Alert: Unregistered individual detected and photographed.
Warning: Unknown person in monitored area. Image captured.
Attention: Unidentified person detected. Security protocols activated.
Alert: Unknown individual detected. Photo saved for review.
Notice: Unrecognized person in area. Security image captured.
Warning: Unknown person detected. Monitoring and recording.
EOF

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Test the wake word detection with: python3 test_wake_word.py"
echo "2. Capture training photos with: python3 image_capture.py"
echo "3. Train the face recognition model with: python3 model_training.py" 
echo "4. Run the enhanced AI assistant with: python3 enhanced_chatty_ai.py"
echo ""
echo "Remember to:"
echo "- Update PERSON_NAME in image_capture.py before taking photos"
echo "- Ensure your camera and microphone permissions are set correctly"
echo "- Test audio playback through your Bluetooth speaker"
echo ""
echo "Press ESC to quit the enhanced AI assistant when running"