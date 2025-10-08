import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import imagehash
from pathlib import Path
import argparse
import csv

# ---------------------------
# Helper Functions
# ---------------------------

def is_blurry(image, threshold=100.0):
    """Detect if an image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def detect_faces_and_smiles(image, face_cascade, smile_cascade):
    """Return number of faces and smiles detected, plus face coordinates."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    smile_count = 0
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
        smile_count += len(smiles)
    
    return len(faces), smile_count, faces

def enhance_image(image):
    """Apply auto enhancements."""
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.1)
    
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def add_watermark(image, watermark_path="watermark.png", position="bottom-right", scale=0.2):
    """Overlay watermark on image."""
    if not os.path.exists(watermark_path):
        return image
    
    base = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    watermark = Image.open(watermark_path).convert("RGBA")

    w_ratio = scale * base.width / watermark.width
    new_size = (int(watermark.width * w_ratio), int(watermark.height * w_ratio))
    watermark = watermark.resize(new_size, Image.ANTIALIAS)
    
    if position == "bottom-right":
        pos = (base.width - watermark.width - 10, base.height - watermark.height - 10)
    else:
        pos = (10, 10)
    
    base.paste(watermark, pos, watermark)
    return cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)

def crop_to_faces(image, faces, margin=0.3):
    """Crop image centered on all detected faces."""
    if len(faces) == 0:
        return image
    
    x_min = min([x for (x, y, w, h) in faces])
    y_min = min([y for (x, y, w, h) in faces])
    x_max = max([x + w for (x, y, w, h) in faces])
    y_max = max([y + h for (x, y, w, h) in faces])
    
    # Add margin
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(int(x_min - width * margin), 0)
    y_min = max(int(y_min - height * margin), 0)
    x_max = min(int(x_max + width * margin), image.shape[1])
    y_max = min(int(y_max + height * margin), image.shape[0])
    
    return image[y_min:y_max, x_min:x_max]

# ---------------------------
# Main Processing
# ---------------------------

def process_folder(input_folder, output_folder, watermark_path=None):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    hashes = set()
    log_data = []

    for file_name in os.listdir(input_folder):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        file_path = os.path.join(input_folder, file_name)
        image = cv2.imread(file_path)
        if image is None:
            continue

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # -----------------
        # Cull checks
        # -----------------
        blurry, blur_score = is_blurry(pil_image)
        hash_val = imagehash.phash(pil_image)
        duplicate = hash_val in hashes
        hashes.add(hash_val)
        face_count, smile_count, faces = detect_faces_and_smiles(pil_image, face_cascade, smile_cascade)

        # Determine cull reason
        cull_reason = ""
        if blurry:
            cull_reason = f"blurry ({blur_score:.2f})"
        elif duplicate:
            cull_reason = "duplicate"
        elif face_count == 0:
            cull_reason = "no face detected"
        else:
            cull_reason = "ok"

        # -----------------
        # Enhancement + Cropping
        # -----------------
        if cull_reason == "ok":
            image = crop_to_faces(image, faces)
            image = enhance_image(image)
            if watermark_path:
                image = add_watermark(image, watermark_path)
            
            out_path = os.path.join(output_folder, file_name)
            cv2.imwrite(out_path, image)

        # -----------------
        # Logging
        # -----------------
        log_data.append({
            "filename": file_name,
            "cull_reason": cull_reason,
            "blur_score": f"{blur_score:.2f}",
            "face_count": face_count,
            "smile_count": smile_count
        })

    # Save log
    log_file = os.path.join(output_folder, "processing_log.csv")
    with open(log_file, mode='w', newline='') as csv_file:
        fieldnames = ["filename", "cull_reason", "blur_score", "face_count", "smile_count"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_data:
            writer.writerow(row)
    
    print(f"Processing complete. Processed images saved to '{output_folder}'.")

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-cull and enhance event photos with smile detection")
    parser.add_argument("--input", required=True, help="Input folder with images")
    parser.add_argument("--output", required=True, help="Output folder for processed images")
    parser.add_argument("--watermark", required=False, help="Path to watermark image (optional)")
    
    args = parser.parse_args()
    process_folder(args.input, args.output, args.watermark)
