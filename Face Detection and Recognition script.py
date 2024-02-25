from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
from retinaface import RetinaFace
import face_recognition
import os
import gc
import numpy as np


def normalize_image(image_path):
    """
    Normalize an image by applying histogram equalization to its luminance channel.
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert the image from RGB to YCrCb color space
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Split the image into Y, Cr, and Cb channels
    y, cr, cb = cv2.split(img_y_cr_cb)
    
    # Apply histogram equalization to the Y channel
    y_eq = cv2.equalizeHist(y)
    
    # Merge the equalized Y channel back with the Cr and Cb channels
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    
    # Convert the image back to RGB color space
    img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2BGR)
    
    return img_rgb_eq

def process_image(image_path, target_encoding, export_dir, enable_target_face):
    try:
        filename = os.path.basename(image_path)
        print(f"Processing file: {filename}")
        
        # Read the original image for drawing and saving
        img = cv2.imread(image_path)
        
        # Create a normalized version of the image for face recognition
        img_normalized = normalize_image(image_path)

        faces = RetinaFace.detect_faces(image_path)  # Face detection on the original image
        match_found = False  # Flag to indicate if a match is found

        if faces:
            for face in faces.values():
                facial_area = face['facial_area']
                x, y, w, h = facial_area
                box_color = (255, 0, 0)  # Default to blue (no match)

                if enable_target_face:
                    face_location = [(y, w, h, x)]  # Convert to face_recognition format
                    # Use the normalized image for face encoding and comparison
                    face_encodings = face_recognition.face_encodings(img_normalized, known_face_locations=face_location)
                    if face_encodings:
                        match = face_recognition.compare_faces([target_encoding], face_encodings[0], tolerance=0.6)[0]
                        if match:
                            box_color = (0, 255, 0)  # Change to green if match
                            match_found = True  # Set flag to True if a match is found

                # Draw rectangles on the original image
                thickness = max(1, int(min(img.shape[:2]) / 200))
                cv2.rectangle(img, (x, y), (w, h), box_color, thickness)

            if match_found:
                new_filename = f"green_detected_{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}"
            else:
                new_filename = f"detected_{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}"
            export_image_path = os.path.join(export_dir, new_filename)
            # Save the original image with rectangles drawn
            cv2.imwrite(export_image_path, img)
            print(f"Processed and saved: {export_image_path}")
        else:
            print(f"No faces detected in {filename}.")
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

def main(input_dir, export_dir, target_face_path, enable_target_face):
    target_image = face_recognition.load_image_file(target_face_path)
    target_encoding = face_recognition.face_encodings(target_image)[0]

    image_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_image, image_path, target_encoding, export_dir, enable_target_face) for image_path in image_paths]
        for future in as_completed(futures):
            future.result()

    gc.collect()

if __name__ == "__main__":
    input_dir = r"input_path"
    export_dir = r"export_path"
    target_face_path = r"Target_Face_Image_Path"
    enable_target_face = True  # Set this to False to disable target face detection
    main(input_dir, export_dir, target_face_path, enable_target_face)
