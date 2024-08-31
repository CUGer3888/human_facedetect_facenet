# from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from PIL import Image
img = cv2.imread("test.jpg")
# detector = MTCNN()
# face = detector.detect_faces(img)
import math
face = [{'box': [334, 529, 801, 1000],
  'confidence': 0.9997301697731018,
  'keypoints': {'left_eye':    (568, 900),
                'right_eye':   (936, 919),
                'nose':        (747, 1075),
                'mouth_left':  (585, 1280),
                'mouth_right': (872, 1302)}}]
# print("face:",face)
bbox,land_mask = face[0]['box'],face[0]['keypoints']
print("bbox:",bbox)
# print("land_mask:",land_mask)
left_eye = [land_mask['left_eye'][0], land_mask['left_eye'][1]]
right_eye = [land_mask['right_eye'][0], land_mask['right_eye'][1]]
k=(right_eye[1]-left_eye[1])/(right_eye[0]-left_eye[0])
print(k)
angle = math.degrees(math.atan(k))
print(angle)
img =Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
img = img.rotate(angle)
img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv2.imshow("img",img)
cv2.waitKey(0)
