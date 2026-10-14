# ===============================================
# TEST SCRIPT: INDIVIDUAL MODULE TESTING
# Fungsi: Test setiap module secara terpisah untuk debugging
# ===============================================
# 
# Penggunaan:
# python test_modules.py
# 
# Menu:
# 1. Test Face Detector
# 2. Test Liveness Detector
# 3. Test Face Recognizer
# 4. Test All Together
#
# ===============================================

import cv2
import numpy as np
import sys
import os
import config
from modules import (
    FaceDetector,
    LivenessDetector,
    FaceRecognizer,
    FPSCounter,
    DisplayHelper,
    log_message
)


def test_face_detector():
    """Test Face Detector module."""
    
    print("\n" + "="*60)
    print("TEST: FACE DETECTOR")
    print("="*60 + "\n")
    
    face_detector = FaceDetector()
    camera = cv2.VideoCapture(config.CAMERA_INDEX)
    fps_counter = FPSCounter()
    
    print("Membuka webcam... Tekan 'q' untuk quit, 's' untuk save frame\n")
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            fps_counter.update()
            fps = fps_counter.get_fps()
            
            # Detect faces
            faces, annotated_frame = face_detector.detect_faces(frame)
            
            # Draw FPS
            annotated_frame = DisplayHelper.draw_fps(annotated_frame, fps)
            
            # Draw info
            info = f"Faces detected: {len(faces)}"
            annotated_frame = DisplayHelper.put_text(
                annotated_frame, info, (10, 40),
                color=config.COLOR_GREEN
            )
            
            # Draw landmarks untuk setiap wajah
            for i, face in enumerate(faces):
                landmarks = face['landmarks']
                bbox = face['bbox']
                
                # Draw bbox
                x, y, w, h = bbox
                annotated_frame = cv2.rectangle(
                    annotated_frame, (x, y), (x+w, y+h),
                    config.COLOR_GREEN, 2
                )
                
                # Draw some landmarks
                for j in range(0, len(landmarks), 10):
                    x, y = int(landmarks[j, 0]), int(landmarks[j, 1])
                    cv2.circle(annotated_frame, (x, y), 2, config.COLOR_BLUE, -1)
                
                # Draw face number
                cv2.putText(annotated_frame, f"Face {i}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_GREEN, 1)
            
            # Display
            cv2.imshow("Test Face Detector", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("test_face_detector.jpg", annotated_frame)
                log_message("Saved: test_face_detector.jpg", "INFO")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()


def test_liveness_detector():
    """Test Liveness Detector module."""
    
    print("\n" + "="*60)
    print("TEST: LIVENESS DETECTOR (BLINK DETECTION)")
    print("="*60 + "\n")
    
    face_detector = FaceDetector()
    liveness_detector = LivenessDetector()
    camera = cv2.VideoCapture(config.CAMERA_INDEX)
    fps_counter = FPSCounter()
    
    print("Membuka webcam... Tekan 'q' untuk quit, 'r' untuk reset\n")
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            fps_counter.update()
            fps = fps_counter.get_fps()
            
            # Detect faces
            faces, annotated_frame = face_detector.detect_faces(frame)
            
            # Draw FPS
            annotated_frame = DisplayHelper.draw_fps(annotated_frame, fps)
            
            if len(faces) > 0:
                face = faces[0]
                landmarks = face['landmarks']
                bbox = face['bbox']
                
                # Get eye landmarks
                left_eye = face_detector.get_eye_landmarks(landmarks, "left")
                right_eye = face_detector.get_eye_landmarks(landmarks, "right")
                
                # Calculate EAR
                ear_left = liveness_detector.calculate_ear(left_eye)
                ear_right = liveness_detector.calculate_ear(right_eye)
                
                # Detect blink
                blink_detected = liveness_detector.detect_blink(ear_left, ear_right)
                liveness_detector.update_liveness(blink_detected)
                
                # Draw eye landmarks
                annotated_frame = face_detector.draw_eye_landmarks(
                    annotated_frame, left_eye, config.COLOR_YELLOW
                )
                annotated_frame = face_detector.draw_eye_landmarks(
                    annotated_frame, right_eye, config.COLOR_YELLOW
                )
                
                # Draw bbox
                x, y, w, h = bbox
                annotated_frame = cv2.rectangle(
                    annotated_frame, (x, y), (x+w, y+h),
                    config.COLOR_GREEN, 2
                )
                
                # Display EAR values
                ear_text = f"EAR L: {ear_left:.4f} | R: {ear_right:.4f} | Avg: {(ear_left+ear_right)/2:.4f}"
                annotated_frame = DisplayHelper.put_text(
                    annotated_frame, ear_text, (10, 40),
                    color=config.COLOR_WHITE
                )
                
                # Display threshold
                threshold_text = f"Threshold: {config.EAR_THRESHOLD:.4f}"
                annotated_frame = DisplayHelper.put_text(
                    annotated_frame, threshold_text, (10, 70),
                    color=config.COLOR_WHITE
                )
                
                # Display liveness status
                liveness_text, color = liveness_detector.get_liveness_text()
                annotated_frame = DisplayHelper.put_text(
                    annotated_frame, liveness_text, (10, 100),
                    color=color,
                    bg_color=config.COLOR_BLACK
                )
                
                # Display blink info
                status = liveness_detector.get_status()
                blink_info = f"Blinks: {status['blink_count']} | Frame Counter: {status['frame_counter']}"
                annotated_frame = DisplayHelper.put_text(
                    annotated_frame, blink_info, (10, 130),
                    color=config.COLOR_WHITE
                )
            
            # Display
            cv2.imshow("Test Liveness Detector", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                liveness_detector.reset()
                log_message("Liveness detector reset", "INFO")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()


def test_face_recognizer():
    """Test Face Recognizer module."""
    
    print("\n" + "="*60)
    print("TEST: FACE RECOGNIZER")
    print("="*60 + "\n")
    
    recognizer = FaceRecognizer()
    
    # Print database info
    db_info = recognizer.get_database_info()
    print(f"Database Path: {db_info['database_path']}")
    print(f"Total Persons: {db_info['total_persons']}")
    print(f"Total Photos: {db_info['total_photos']}")
    print("\nPerson breakdown:")
    for person, count in db_info['person_counts'].items():
        print(f"  - {person}: {count} photo(s)")
    
    if recognizer.is_database_empty():
        log_message("Database kosong! Silakan masukkan foto ke face_database/", "WARNING")
        return
    
    print("\nTest akan membuka webcam dan try face recognition")
    print("Tekan 'q' untuk quit\n")
    
    face_detector = FaceDetector()
    camera = cv2.VideoCapture(config.CAMERA_INDEX)
    fps_counter = FPSCounter()
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            fps_counter.update()
            fps = fps_counter.get_fps()
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces, annotated_frame = face_detector.detect_faces(frame)
            
            # Draw FPS
            annotated_frame = DisplayHelper.draw_fps(annotated_frame, fps)
            
            if len(faces) > 0:
                face = faces[0]
                bbox = face['bbox']
                
                # Encode face
                face_location = (bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0])
                face_encoding = recognizer.encode_face(frame_rgb, face_location)
                
                # Draw bbox
                x, y, w, h = bbox
                annotated_frame = cv2.rectangle(
                    annotated_frame, (x, y), (x+w, y+h),
                    config.COLOR_GREEN, 2
                )
                
                if face_encoding is not None:
                    # Recognize face
                    person_name, confidence = recognizer.recognize_face(face_encoding)
                    
                    # Get access result
                    access_text, color = recognizer.get_access_result(person_name)
                    
                    # Display result
                    result_text = f"{person_name} ({confidence:.2%})"
                    annotated_frame = DisplayHelper.put_text(
                        annotated_frame, result_text, (10, 40),
                        color=color
                    )
                    
                    annotated_frame = DisplayHelper.put_text(
                        annotated_frame, access_text, (10, 70),
                        color=color,
                        bg_color=config.COLOR_BLACK
                    )
            
            # Display
            cv2.imshow("Test Face Recognizer", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    """Main menu."""
    
    print("\n" + "="*60)
    print("MODULE TEST MENU")
    print("="*60 + "\n")
    
    while True:
        print("\nPilihan:")
        print("1. Test Face Detector")
        print("2. Test Liveness Detector")
        print("3. Test Face Recognizer")
        print("4. Exit")
        print()
        
        choice = input("Pilih (1-4): ").strip()
        
        if choice == "1":
            test_face_detector()
        elif choice == "2":
            test_liveness_detector()
        elif choice == "3":
            test_face_recognizer()
        elif choice == "4":
            log_message("Exit", "INFO")
            break
        else:
            log_message("Pilihan tidak valid", "WARNING")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_message("Program diinterupsi", "INFO")
