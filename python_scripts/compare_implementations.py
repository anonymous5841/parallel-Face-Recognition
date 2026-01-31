import os
import sys
import urllib.parse
from face_recognition_serial import process_images_serial
from face_recognition_accurate import process_images_parallel


def run_parallel_version(reference_folder, test_image):
    print("\nRunning Parallel Version:")
    return process_images_parallel(reference_folder, test_image)


def run_serial_version(reference_folder, test_image):
    print("\nRunning Serial Version:")
    return process_images_serial(reference_folder, test_image)


def print_results(version_name, results, execution_time):
    print(f"\n{version_name} Results:")
    print(f"Execution time: {execution_time:.2f} seconds")
    for image_path, match, distance, error in results:
        if error:
            print(f"Error processing {image_path}: {error}")
        elif match:
            print(f"MATCH: {image_path} matches {match} (score: {1/(1+distance):.2%})")
        else:
            print(f"NO MATCH: {image_path} (best score: {1/(1+distance):.2%})")


def compare_implementations(img):
    """Compare serial and parallel implementations for a single image path."""
    reference_folder = "reference_faces"

    if not img or not os.path.exists(img):
        print("No valid test image specified.")
        return

    print(f"Using test image: {img}")

    serial_results, serial_time = run_serial_version(reference_folder, img)
    parallel_results, parallel_time = run_parallel_version(reference_folder, img)

    print("\n" + "=" * 40)
    print_results("Serial Version", serial_results, serial_time)
    print_results("Parallel Version", parallel_results, parallel_time)

    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    print("\nPerformance Summary:")
    print(f"Serial:   {serial_time:.2f}s")
    print(f"Parallel: {parallel_time:.2f}s")
    print(f"Speedup:  {speedup:.2f}x")

    if len(serial_results) == len(parallel_results):
        matches = sum(1 for sr, pr in zip(serial_results, parallel_results) if sr[1] == pr[1])
        print(f"Result Consistency: {matches}/{len(serial_results)}")
    else:
        print("Warning: serial and parallel produced different result counts")


if __name__ == "__main__":
    # Accept an image path (filesystem or web-style) as a CLI arg.
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    img = None
    if arg:
        arg = urllib.parse.unquote(arg)
        if arg.startswith("/") or "temp_recognize" in arg:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            candidate = os.path.join(project_root, arg.lstrip("/\\"))
            if os.path.exists(candidate):
                img = candidate
            else:
                candidate2 = os.path.join(os.path.dirname(__file__), arg.lstrip("/\\"))
                img = candidate2 if os.path.exists(candidate2) else candidate
        else:
            img = os.path.abspath(os.path.expanduser(arg))

    compare_implementations(img)