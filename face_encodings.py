#!/usr/bin/env python3
"""
Create an empty face encodings file to eliminate the warning
"""

import pickle
import os

def create_empty_face_encodings():
    """Create an empty face encodings file"""
    data = {
        'encodings': [],
        'names': []
    }
    
    with open('face_encodings.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print("✅ Created empty face_encodings.pkl file")
    print("This eliminates the 'No face encodings file found' warning")
    print("You can add known faces later if needed")

if __name__ == "__main__":
    create_empty_face_encodings()