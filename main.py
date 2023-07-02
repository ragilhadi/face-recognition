import cv2 as cv
import numpy as np
from utils.face_recognition import FaceRecognition

FR = FaceRecognition()

capture = cv.VideoCapture(0)

while True:
    _, frame = capture.read()
    frame = cv.flip(frame, 1)
    frame = FR.face_detection(frame)
    frame = FR.draw_points(frame)
    cv.imshow('Video Capture',frame)

    if cv.waitKey(20) & 0XFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()