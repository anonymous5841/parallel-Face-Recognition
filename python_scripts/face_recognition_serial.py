import cv2
import numpy as np
import os
import time
from face_utils import preprocess_face, generate_rotations, compute_lbp_hist_from_gray, compute_simple_hog


class LBPFaceRecognizerSerial:
    def __init__(self, reference_folder):
        self.reference_folder = reference_folder
        self.reference_features = {}

    def load_reference_images(self):
        if not os.path.exists(self.reference_folder):
            raise ValueError(f"Reference folder '{self.reference_folder}' does not exist")

        image_files = [f for f in os.listdir(self.reference_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif'))]

        for image_file in image_files:
            image_path = os.path.join(self.reference_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            pre = preprocess_face(image)
            if pre is None:
                continue
            gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
            lbp_hist = compute_lbp_hist_from_gray(gray)
            hog = compute_simple_hog(gray)
            self.reference_features[image_file] = {'lbp': lbp_hist, 'hog': hog}

    def compare_features(self, hist1, hist2):
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

    def recognize_face(self, input_image, threshold=0.1, weights=(0.7, 0.3)):
        if not self.reference_features:
            print("Warning: No reference images loaded")
            return None, float('inf')

        pre = preprocess_face(input_image)
        if pre is None:
            return None, float('inf')

        rotations = generate_rotations(pre, angles=(0, 15, -15))
        best_match = None
        best_similarity = 0.0

        for rot in rotations:
            gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
            input_lbp = compute_lbp_hist_from_gray(gray)
            input_hog = compute_simple_hog(gray)

            for image_name, ref in self.reference_features.items():
                chi = cv2.compareHist(input_lbp, ref['lbp'], cv2.HISTCMP_CHISQR)
                hog_dist = float(np.linalg.norm(input_hog - ref['hog']))
                s_lbp = 1.0 / (1.0 + chi)
                s_hog = 1.0 / (1.0 + hog_dist)
                combined_sim = weights[0] * s_lbp + weights[1] * s_hog

                if combined_sim > best_similarity:
                    best_similarity = combined_sim
                    best_match = image_name

        if best_similarity <= 0:
            return None, float('inf')

        combined_distance = (1.0 / best_similarity) - 1.0
        if combined_distance < threshold:
            return best_match, combined_distance
        return None, combined_distance


def process_images_serial(reference_folder, test_image):
    results = []
    start_time = time.time()

    recognizer = LBPFaceRecognizerSerial(reference_folder)
    print("Loading reference images...")
    recognizer.load_reference_images()

    if not test_image or not os.path.exists(test_image):
        print(f"Test image '{test_image}' does not exist")
        return results, 0

    image_path = test_image
    try:
        image = cv2.imread(image_path)
        if image is None:
            results.append((image_path, None, None, "Failed to load image"))
        else:
            match, distance = recognizer.recognize_face(image)
            results.append((image_path, match, distance, None))
    except Exception as e:
        results.append((image_path, None, None, str(e)))

    execution_time = time.time() - start_time
    return results, execution_time


if __name__ == "__main__":
    reference_folder = "reference_faces"
    test_image = None

    if test_image is None:
        print("Set `test_image` to a valid image path to run this module directly.")
    else:
        results, execution_time = process_images_serial(reference_folder, test_image)
        print(f"\nExecution time: {execution_time:.2f} seconds")
        for image_path, match, distance, error in results:
            if error:
                print(f"Error processing {image_path}: {error}")
            elif match:
                print(f"MATCH: {image_path} matches {match} (score: {1/(1+distance):.2%})")
            else:
                print(f"NO MATCH: {image_path} (best score: {1/(1+distance):.2%})")