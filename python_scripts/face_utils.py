import cv2
import numpy as np
import math


def preprocess_face(image, target_size=(128, 128)):
    """Detect face, align eyes when possible, equalize lighting, and resize."""
    if image is None:
        return None

    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    x = y = w = h = None
    if len(faces) > 0:
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        x, y, w, h = faces[0]

        face_roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray)
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                cx = x + ex + ew // 2
                cy = y + ey + eh // 2
                eye_centers.append((cx, cy))
            (x1, y1), (x2, y2) = eye_centers[0], eye_centers[1]
            if x2 != x1:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                center = (img.shape[1] // 2, img.shape[0] // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
                    x, y, w, h = faces[0]

    if x is None:
        h_img, w_img = img.shape[:2]
        side = min(w_img, h_img)
        cx, cy = w_img // 2, h_img // 2
        x = max(0, cx - side // 2)
        y = max(0, cy - side // 2)
        w = h = side

    pad_x = int(0.1 * w)
    pad_y = int(0.15 * h)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.shape[1], x + w + pad_x)
    y2 = min(img.shape[0], y + h + pad_y)

    face = img[y1:y2, x1:x2]
    try:
        ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        y_channel, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y_channel)
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        face = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    except Exception:
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray_face)
        face = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
    return face


def rotate_image(image, angle):
    if image is None:
        return None
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)


def generate_rotations(image, angles=(0, 15, -15)):
    return [rotate_image(image, a) for a in angles]


def compute_lbp_hist_from_gray(gray):
    if gray is None:
        return None
    height, width = gray.shape
    if height < 3 or width < 3:
        return np.zeros((1, 256), dtype=np.float32)

    lbp = np.zeros((height - 2, width - 2), dtype=np.uint8)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            center = gray[i, j]
            code = 0
            code |= (gray[i-1, j-1] >= center) << 7
            code |= (gray[i-1, j] >= center) << 6
            code |= (gray[i-1, j+1] >= center) << 5
            code |= (gray[i, j+1] >= center) << 4
            code |= (gray[i+1, j+1] >= center) << 3
            code |= (gray[i+1, j] >= center) << 2
            code |= (gray[i+1, j-1] >= center) << 1
            code |= (gray[i, j-1] >= center) << 0
            lbp[i-1, j-1] = code

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist.reshape(1, -1)


def compute_simple_hog(gray, bins=9):
    if gray is None:
        return np.zeros((bins,), dtype=np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = np.mod(ang, 180.0)

    bin_width = 180.0 / bins
    hist = np.zeros((bins,), dtype=np.float32)
    for b in range(bins):
        low = b * bin_width
        high = (b + 1) * bin_width
        mask = (ang >= low) & (ang < high)
        hist[b] = np.sum(mag[mask])

    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist
