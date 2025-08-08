#!/bin/bash
# Fix Chatty AI Model Paths and Configuration
echo "🔧 Fixing Chatty AI Model Paths and Dependencies"
echo "================================================="

# Check current directory
if [ ! -f "app.py" ]; then
    echo "❌ Please run this from the Chatty_AI directory"
    exit 1
fi

echo "1. Current directory structure:"
ls -la | grep -E "(models|tinyllama)"

echo -e "\n2. Checking tinyllama-models directory:"
if [ -d "tinyllama-models" ]; then
    echo "✅ tinyllama-models directory found"
    ls -la tinyllama-models/
else
    echo "❌ tinyllama-models directory not found"
fi

echo -e "\n3. Creating symbolic link for models directory..."
# Create a symbolic link from models to tinyllama-models
if [ -d "tinyllama-models" ] && [ ! -d "models" ]; then
    ln -s tinyllama-models models
    echo "✅ Created symbolic link: models -> tinyllama-models"
elif [ -d "models" ]; then
    echo "✅ Models directory already exists"
else
    echo "❌ Cannot create models link - tinyllama-models not found"
fi

echo -e "\n4. Verifying model file access..."
MODEL_FILE="tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf"
if [ -f "$MODEL_FILE" ]; then
    echo "✅ TinyLlama model file found"
    echo "   File: $MODEL_FILE"
    echo "   Size: $(du -sh "$MODEL_FILE" | cut -f1)"
else
    echo "❌ TinyLlama model file not found at: $MODEL_FILE"
fi

echo -e "\n5. Checking for configuration files that might need updating..."
# Check if there are any config files that hardcode the models path
for config_file in config.py settings.py chatty_ai.py app.py; do
    if [ -f "$config_file" ]; then
        if grep -q "models/" "$config_file"; then
            echo "Found 'models/' references in $config_file:"
            grep -n "models/" "$config_file" | head -3
        fi
    fi
done

echo -e "\n6. Installing missing Python packages (optional for better performance)..."
echo "   Note: PyTorch and Transformers are not required for GGUF models"
echo "   But they can provide additional functionality."

# Check if user wants to install PyTorch (it's large)
echo -e "\n   Would you like to install PyTorch? (This will take ~500MB space and time)"
echo "   Press 'y' for yes, any other key to skip:"
read -t 10 -n 1 response
echo

if [[ "$response" = "y" || "$response" = "Y" ]]; then
    echo "Installing PyTorch (this may take several minutes)..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    
    echo "Installing Transformers..."
    pip install transformers
else
    echo "Skipping PyTorch installation - GGUF models will work without it"
fi

echo -e "\n7. Testing the fix..."
python3 -c "
import os
print('Checking model paths:')
print('- tinyllama-models exists:', os.path.exists('tinyllama-models'))
print('- models symlink exists:', os.path.exists('models'))
print('- model file exists:', os.path.exists('tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf'))
"

echo -e "\n✅ Model path fix completed!"
echo -e "\n🚀 Next steps:"
echo "1. Start the application: python3 app.py"
echo "2. Open browser to: http://192.168.1.16:5000"
echo "3. Click 'Start System' - it should work now!"
echo ""
echo "💡 The system uses GGUF models which don't require PyTorch"
echo "   This is actually more efficient for Raspberry Pi!"