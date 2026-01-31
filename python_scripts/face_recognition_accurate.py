import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
from face_utils import preprocess_face, generate_rotations, compute_lbp_hist_from_gray, compute_simple_hog
class LBPFaceRecognizerAccurate:
    def __init__(self, reference_folder):
        self.reference_folder = reference_folder
        self.reference_features = {}
        
    @staticmethod
    def calculate_lbp_vectorized(gray):
        """Calculate LBP features using NumPy vectorization."""
        height, width = gray.shape
        lbp = np.zeros((height-2, width-2), dtype=np.uint8)
        
        # Get all 3x3 windows at once
        windows = np.lib.stride_tricks.sliding_window_view(gray, (3, 3))
        centers = windows[:, :, 1, 1]
        
        # Calculate LBP using broadcasting
        powers = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
        neighbors = [
            windows[:, :, 0, 0], windows[:, :, 0, 1], windows[:, :, 0, 2],
            windows[:, :, 1, 2], windows[:, :, 2, 2], windows[:, :, 2, 1],
            windows[:, :, 2, 0], windows[:, :, 1, 0]
        ]
        
        for power, neighbor in zip(powers, neighbors):
            lbp += power * (neighbor >= centers)
        
        # Calculate normalized histogram
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist = hist.astype(np.float32)
        hist = hist / hist.sum()
        
        return hist

    def load_reference_images(self):
        if not os.path.exists(self.reference_folder):
            raise ValueError(f"Reference folder '{self.reference_folder}' does not exist")

        image_files = [f for f in os.listdir(self.reference_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif'))]

        def process_image(image_file):
            image_path = os.path.join(self.reference_folder, image_file)
            img = cv2.imread(image_path)
            if img is None:
                return None
            pre = preprocess_face(img)
            if pre is None:
                return None
            gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
            lbp = self.calculate_lbp_vectorized(gray)
            hog = compute_simple_hog(gray)
            return (image_file, {'lbp': lbp, 'hog': hog})

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_image, image_files))

        self.reference_features = {name: feats for name, feats in filter(None, results)}

def compare_features_accurate(input_feats, ref_feats, weights=(0.7, 0.3)):
    """Compare LBP+HOG features and return a distance-like value."""
    chi = cv2.compareHist(input_feats['lbp'], ref_feats['lbp'], cv2.HISTCMP_CHISQR)
    hog_dist = float(np.linalg.norm(input_feats['hog'] - ref_feats['hog']))
    s_lbp = 1.0 / (1.0 + chi)
    s_hog = 1.0 / (1.0 + hog_dist)
    combined_sim = weights[0] * s_lbp + weights[1] * s_hog
    if combined_sim <= 0:
        return float('inf')
    return (1.0 / combined_sim) - 1.0

def process_single_image(args):
    image_path, reference_features, threshold = args

    img = cv2.imread(image_path)
    if img is None:
        return image_path, None, None, "Failed to load image"

    pre = preprocess_face(img)
    if pre is None:
        return image_path, None, None, "Failed to preprocess image"

    rotations = generate_rotations(pre, angles=(0, 15, -15))
    ref_items = list(reference_features.items())

    best_match = None
    best_distance = float('inf')

    for rot in rotations:
        gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
        input_lbp = LBPFaceRecognizerAccurate.calculate_lbp_vectorized(gray)
        input_hog = compute_simple_hog(gray)
        input_feats = {'lbp': input_lbp, 'hog': input_hog}

        if len(ref_items) >= 4:
            chunk_size = max(1, len(ref_items) // (os.cpu_count() or 1))
            chunks = [ref_items[i:i + chunk_size] for i in range(0, len(ref_items), chunk_size)]
            with ThreadPoolExecutor() as executor:
                def compare_chunk(ref_chunk):
                    return [(name, compare_features_accurate(input_feats, features)) for name, features in ref_chunk]
                chunk_results = list(executor.map(compare_chunk, chunks))
            all_results = [item for sublist in chunk_results for item in sublist]
        else:
            all_results = [(name, compare_features_accurate(input_feats, features)) for name, features in ref_items]

        for name, dist in all_results:
            if dist < best_distance:
                best_distance = dist
                best_match = name

    if best_distance < threshold:
        return image_path, best_match, best_distance, None
    return image_path, None, best_distance, None

def process_images_parallel(reference_folder, test_image):
    """Process a single image with parallel feature comparison"""
    start_time = time.time()

    # Initialize and load reference images
    recognizer = LBPFaceRecognizerAccurate(reference_folder)
    print("Loading reference images...")
    recognizer.load_reference_images()

    if not test_image or not os.path.exists(test_image):
        print(f"Test image '{test_image}' does not exist")
        return [], 0

    # Process image
    threshold = 0.1
    process_args = [(test_image, recognizer.reference_features, threshold)]

    # Use parallel processing internally for comparing references; only one input image
    results = [process_single_image(args) for args in process_args]

    execution_time = time.time() - start_time
    return results, execution_time

if __name__ == "__main__":
    reference_folder = "reference_faces"
    # Set `test_image` to a full path of a single image to test, e.g.:
    # test_image = r"C:\path\to\image.jpg"
    test_image = None

    if test_image is None:
        print("Set `test_image` to a valid image path to run this module directly.")
    else:
        results, execution_time = process_images_parallel(reference_folder, test_image)

        print(f"\nExecution time: {execution_time:.2f} seconds")
        for image_path, match, distance, error in results:
            if error:
                print(f"Error processing {image_path}: {error}")
            elif match:
                print(f"MATCH FOUND: {image_path} matches with {match} "
                      f"(similarity score: {1/(1+distance):.2%})")
            else:
                print(f"NO MATCH: {image_path} does not match any reference image "
                      f"(best similarity score: {1/(1+distance):.2%})")