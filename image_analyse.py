from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from picamera2 import Picamera2
import time
from PIL import Image

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize camera (we'll do this lazily to avoid keeping it always on)
picam2 = None

def initialize_camera():
    """Initialize the camera if not already done"""
    global picam2
    if picam2 is None:
        try:
            picam2 = Picamera2()
            # Configure for still capture
            still_config = picam2.create_still_configuration(
                main={"size": (1640, 1232)},  # Good balance of quality and file size
                lores={"size": (640, 480)},   # Lower resolution for preview
                display="lores"
            )
            picam2.configure(still_config)
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    return True

def capture_image():
    """Capture an image and return it as base64"""
    global picam2
    
    if not initialize_camera():
        return None
    
    try:
        # Start camera if not running
        if not picam2.started:
            picam2.start()
            time.sleep(2)  # Allow camera to stabilize
        
        # Capture image to memory
        image_array = picam2.capture_array()
        
        # Convert to PIL Image and then to base64
        image = Image.fromarray(image_array)
        
        # Convert to JPEG in memory
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_base64
        
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None

def cleanup_camera():
    """Clean up camera resources"""
    global picam2
    if picam2 and picam2.started:
        picam2.stop()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        include_camera = request.json.get('include_camera', False)
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Prepare the messages array
        messages = []
        
        if include_camera:
            # Capture image
            image_base64 = capture_image()
            
            if image_base64:
                # Create message with image
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_message
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                })
            else:
                # Fallback to text-only if camera fails
                messages.append({
                    "role": "user",
                    "content": user_message + " (Note: Camera capture failed)"
                })
        else:
            # Text-only message
            messages.append({
                "role": "user",
                "content": user_message
            })
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # This model supports vision
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content
        
        return jsonify({
            'response': assistant_message,
            'camera_used': include_camera and image_base64 is not None
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/capture_test')
def capture_test():
    """Test endpoint to capture and return an image"""
    try:
        image_base64 = capture_image()
        if image_base64:
            return f'''
            <html>
                <body>
                    <h2>Camera Test - Image Captured Successfully!</h2>
                    <img src="data:image/jpeg;base64,{image_base64}" style="max-width: 100%; height: auto;">
                    <br><br>
                    <a href="/">Back to Chat</a>
                </body>
            </html>
            '''
        else:
            return '<h2>Camera Test Failed</h2><a href="/">Back to Chat</a>'
    except Exception as e:
        return f'<h2>Error: {str(e)}</h2><a href="/">Back to Chat</a>'

@app.teardown_appcontext
def cleanup(error):
    """Clean up resources when app context tears down"""
    cleanup_camera()

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        cleanup_camera()