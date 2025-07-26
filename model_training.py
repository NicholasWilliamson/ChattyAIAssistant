import os
from imutils import paths
import face_recognition
import pickle
import cv2

print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("dataset"))
knownEncodings = []
knownNames = []

processed_count = 0
failed_count = 0

for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}: {imagePath}")
    name = imagePath.split(os.path.sep)[-2]
    
    try:
        image = cv2.imread(imagePath)
        if image is None:
            print(f"⚠️  Could not read image: {imagePath}")
            failed_count += 1
            continue
            
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Try multiple detection methods for better mask recognition
        boxes_hog = face_recognition.face_locations(rgb, model="hog")
        boxes_cnn = face_recognition.face_locations(rgb, model="cnn") if len(boxes_hog) == 0 else []
        
        # Use the method that found faces
        boxes = boxes_hog if len(boxes_hog) > 0 else boxes_cnn
        
        if len(boxes) == 0:
            print(f"⚠️  No faces found in: {imagePath}")
            failed_count += 1
            continue
            
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        if len(encodings) == 0:
            print(f"⚠️  Could not encode faces in: {imagePath}")
            failed_count += 1
            continue
        
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
            processed_count += 1
            
        print(f"✅ Successfully processed {len(encodings)} face(s) from {imagePath}")
        
    except Exception as e:
        print(f"❌ Error processing {imagePath}: {e}")
        failed_count += 1
        continue

print(f"\n[INFO] Processing complete!")
print(f"Successfully processed: {processed_count} face encodings")
print(f"Failed to process: {failed_count} images")

if processed_count == 0:
    print("❌ No face encodings were created. Check your images.")
    exit(1)

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print(f"[INFO] Training complete. {processed_count} encodings saved to 'encodings.pickle'")

# Print summary by person
from collections import Counter
name_counts = Counter(knownNames)
print("\n[INFO] Encodings per person:")
for name, count in name_counts.items():
    print(f"  {name}: {count} encodings")