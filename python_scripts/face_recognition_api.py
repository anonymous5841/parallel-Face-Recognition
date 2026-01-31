#!/usr/bin/env python3
import cv2
import sys
import json
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from face_recognition_accurate import LBPFaceRecognizerAccurate, process_single_image
from face_recognition_serial import LBPFaceRecognizerSerial

def get_reference_face_image(face_name, reference_folder):
    for ext in ('.jfif', '.jpg', '.jpeg', '.png'):
        path = os.path.join(reference_folder, face_name + ext)
        if os.path.exists(path):
            return path
    return None

def recognize_face(image_path, reference_folder):
    try:
        recognizer_parallel = LBPFaceRecognizerAccurate(reference_folder)
        recognizer_parallel.load_reference_images()

        recognizer_serial = LBPFaceRecognizerSerial(reference_folder)
        recognizer_serial.load_reference_images()

        if not recognizer_parallel.reference_features:
            return {"match": False, "error": "No reference faces loaded", "score": 0.0, "serial_time": 0, "parallel_time": 0, "speedup": 0}

        img = cv2.imread(image_path)
        start_serial = time.time()
        serial_match, serial_dist = recognizer_serial.recognize_face(img, threshold=0.1)
        serial_time = time.time() - start_serial

        start_parallel = time.time()
        result_data = process_single_image((image_path, recognizer_parallel.reference_features, 0.1))
        parallel_time = time.time() - start_parallel

        speedup = serial_time / parallel_time if parallel_time > 0 else 0

        _, best_match, best_distance, error = result_data
        if error:
            return {"match": False, "error": error, "score": 0.0}

        similarity = 1.0 / (1.0 + best_distance) if best_distance != float('inf') else 0.0

        response = {"serial_time": serial_time, "parallel_time": parallel_time, "speedup": speedup, "score": float(similarity), "distance": float(best_distance)}

        if best_match and similarity > 0.5:
            clean_name = best_match.replace('.jfif', '').replace('.jpg', '').replace('.jpeg', '').replace('.png', '').title()
            response.update({"match": True, "name": clean_name, "img": f"/reference_faces/{best_match}"})
        else:
            response.update({"match": False, "name": None, "img": None})

        return response
    except Exception as e:
        return {"match": False, "error": str(e), "score": 0.0}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"match": False, "error": "Usage: face_recognition_api.py <image_path> [reference_folder]"}))
        sys.exit(1)

    raw_path = sys.argv[1]
    reference_folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(current_dir), "reference_faces")

    temp_file_path = None
    if raw_path == "--stdin":
        import tempfile
        data = sys.stdin.buffer.read()
        if not data:
            print(json.dumps({"match": False, "error": "No image data received on stdin"}))
            sys.exit(1)
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=current_dir)
        try:
            tf.write(data)
            tf.flush()
            temp_file_path = tf.name
        finally:
            tf.close()
        raw_path = temp_file_path

    def resolve_path(p):
        p = p or ""
        p = os.path.expanduser(p)
        if p.startswith("/") or "temp_recognize" in p or "reference_faces" in p:
            project_root = os.path.abspath(os.path.join(current_dir, ".."))
            candidate = os.path.join(project_root, p.lstrip("/\\"))
            if os.path.exists(candidate):
                return candidate
            candidate2 = os.path.join(current_dir, p.lstrip("/\\"))
            if os.path.exists(candidate2):
                return candidate2
            return candidate
        if os.path.isabs(p):
            return p
        candidate = os.path.join(current_dir, p)
        if os.path.exists(candidate):
            return candidate
        return os.path.abspath(p)

    image_path = resolve_path(raw_path)
    if not os.path.exists(image_path):
        print(json.dumps({"match": False, "error": f"Image file not found: {image_path}"}))
        sys.exit(1)

    result = recognize_face(image_path, reference_folder)
    import math
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, float):
            return None if not math.isfinite(obj) else obj
        return obj

    safe_result = sanitize(result)
    try:
        print(json.dumps(safe_result))
    except Exception as e:
        print(json.dumps({"match": False, "error": f"Failed to serialize result: {str(e)}"}))

    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
        except Exception:
            pass