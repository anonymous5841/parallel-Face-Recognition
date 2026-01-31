Environment and dependencies

Overview
- This project is a Node.js + Python app. Node/Express serves the frontend and forwards uploaded images to Python for face recognition.

Recommended versions
- Node.js: 18.x or 20.x (LTS: 18.16+/20.x recommended)
- npm: bundled with Node; use the npm that comes with the Node release above
- Python: 3.11+ (tested on Python 3.13)

Python packages
- The Python scripts require OpenCV and NumPy. Install via pip using the included `requirements.txt`.

System notes (Windows)
- On Windows, installing `opencv-python` may require the Microsoft Visual C++ Redistributable.
- Ensure Python is available on PATH so `server.js` can spawn the Python process (it calls `python`). If your Python executable is `python3` or at a custom path, update `server.js` or the environment.

Quick setup
1. Install Node dependencies

   npm install

2. Create and activate a Python virtual environment (recommended)

   python -m venv .venv
   .\.venv\Scripts\activate

3. Install Python requirements

   pip install --upgrade pip
   pip install -r requirements.txt

4. Run the Node server

   npm start

How the pieces map
- Node: `server.js`, `package.json` — serves static pages and calls Python.
- Python: files under `python_scripts/` (face recognition implementations and utils).
- Reference images: `reference_faces/` — must contain reference face images for recognition.

