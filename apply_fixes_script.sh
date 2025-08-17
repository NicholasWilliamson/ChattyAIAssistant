#!/bin/bash

# Chatty AI Fixes Application Script
echo "Applying Chatty AI fixes..."

# 1. Lower the silence threshold
echo "1. Lowering silence threshold for better wake word detection..."
sed -i 's/SILENCE_THRESHOLD = 0.035/SILENCE_THRESHOLD = 0.015/' chatty_ai.py

# 2. Fix the wake word detection threshold in record_wake_word_check
echo "2. Improving wake word detection sensitivity..."
sed -i 's/if rms > SILENCE_THRESHOLD \* 2:/if rms > SILENCE_THRESHOLD * 1.5:/' chatty_ai.py

# 3. Add debug logging to wake word check (add after the rms calculation)
echo "3. Adding debug logging to wake word detection..."
# This will add the debug line after the RMS calculation
sed -i '/rms = np.sqrt(np.mean(audio_data\*\*2))/a\                self.emit_log(f"Wake word audio RMS: {rms:.4f} (threshold: {SILENCE_THRESHOLD * 1.5:.4f})", '"'"'debug'"'"')' chatty_ai.py

# 4. Add logging when audio is saved
sed -i '/sf.write(WAKE_WORD_AUDIO, audio_data, SAMPLE_RATE)/a\                    self.emit_log(f"Wake word audio saved - RMS {rms:.4f} exceeded threshold", '"'"'info'"'"')' chatty_ai.py

# 5. Add logging when audio is too quiet  
sed -i '/return False/i\                    self.emit_log(f"Wake word audio too quiet - RMS {rms:.4f} below threshold", '"'"'debug'"'"')' chatty_ai.py

echo "✅ Wake word detection fixes applied!"
echo ""
echo "Now you need to manually replace the LLM methods. Here are the line numbers to replace:"
echo ""
echo "MANUAL REPLACEMENTS NEEDED:"
echo "=========================="
echo ""
echo "METHOD 1: get_llm_joke() - Starting around line 1020"
echo "- Replace the entire method with the improved version from the artifact above"
echo ""
echo "METHOD 2: get_llm_fun_fact() - Starting around line 1045" 
echo "- Replace the entire method with the improved version from the artifact above"
echo ""
echo "METHOD 3: check_for_bored_response() - Starting at line 1071"
echo "- Replace the entire method with the improved version from the artifact above"
echo ""
echo "To do this manually:"
echo "1. Open chatty_ai.py in your editor"
echo "2. Find each method by line number"
echo "3. Replace the entire method with the version from the artifact"
echo ""
echo "OR use these sed commands for quick method replacement:"

# Provide sed commands for method replacements
cat << 'EOF'

# Replace get_llm_joke method (use carefully - check line numbers first)
echo "Backing up before method replacements..."
cp chatty_ai.py chatty_ai.py.backup_methods

# You can try these sed commands, but it's safer to manually replace:
# sed -i '/def get_llm_joke/,/return.*octopuses.*blue blood/c\[NEW METHOD HERE]' chatty_ai.py

EOF

echo ""
echo "🔧 NEXT STEPS:"
echo "1. The silence threshold has been lowered automatically"
echo "2. Wake word detection logging has been improved" 
echo "3. You need to manually replace the 3 LLM methods using the code from the artifact"
echo "4. Test the system to ensure wake word detection is more responsive"
echo ""
echo "The main improvements:"
echo "- More sensitive wake word detection"
echo "- Better LLM prompts with randomization"
echo "- Proper delays and separate speech segments"
echo "- Enhanced debugging and logging"