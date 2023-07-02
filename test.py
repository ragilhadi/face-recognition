import cv2 as cv
import numpy as np
from utils.face_recognition import FaceRecognition

FR = FaceRecognition()

frame = cv.imread('Dataset/CD.png')

frame = FR.face_detection(frame)
frame = FR.draw_points(frame)
frame, features = FR.draw_line(frame)
frame = FR.knn_predict(features, frame)

cv.imshow('Image',frame)
# cv.imwrite('test/testing/CD.jpg',frame)
cv.waitKey(0)
cv.destroyAllWindows()