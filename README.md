**PARALLEL FACE RECOGNITION USING PYTHON**

Description:
the above code implements the concept of parallel face recognition using LBP(Local Binary Pattern) and HOG(Histogram of Oriented Gradient) for feature extraction with preprocessing step like face alignment, rotation, lightening equalization etc to better the detection of face even with subtle changes. The above code implements both **serial version and parallel version** of face recognition using numpy and cv2 libraries.

The code contains both terminal execution and web based execution options 

**For terminal based execution:**
Option 1:

1.First add image path to the img variable in compare_implementations.py
2.Run the compare_implementation.py file

Option 2:
use command:"c:/Users/(Username)/Desktop/(foldername)/python_scripts/compare_implementations.py" "c:/Users/(Username)/Desktop/(foldername)/reference_faces/aina.jpg"

**For web absed execution:**
Run command in terminal: node server.js
ctrl+click the localhost link it prints in the terminal

**Results/Ouput:**
1.serial time
2.parallel time
3.confidence score
4.speedup
5.Matched img name
in web based extra features:
6:data edit/delete
7:data registeration
