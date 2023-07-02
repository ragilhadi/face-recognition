import cv2 as cv
import numpy as np
from utils.face_recognition import FaceRecognition

FR = FaceRecognition()

frame = cv.imread('Dataset/A.jpeg')

# frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# frame = FR.face_detection(frame)
frame = FR.draw_points(frame)
frame, features = FR.draw_line(frame)


cv.imshow('Image',frame)
cv.imwrite('test/linedistance.jpg',frame)
cv.waitKey(0)
cv.destroyAllWindows()