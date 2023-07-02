import dlib
import cv2 as cv
import numpy as np
import pickle
import time

class FaceRecognition():
    def __init__(self):
        self.file_path_haarcascade = "assets/haarcascade.xml"
        self.classifier = cv.CascadeClassifier(self.file_path_haarcascade)
        self.file_path_faciallandmark = "assets/shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.file_path_faciallandmark)
        self.lk_params = dict(winSize=(70, 70),
                            maxLevel=50,
                            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.003)
                            )
        self.features = []
        self.model_path = 'assets/model.sav'
        self.model = pickle.load(open(self.model_path, 'rb'))

        self.person = ['A', 'B', 'C', 'D']
    
    def face_detection(self, frame):
        img_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces = self.classifier.detectMultiScale(img_gray)
        for (x,y,w,h) in faces:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        return frame
        
    
    def draw_points(self, frame):
        img_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces = self.detector(img_gray, 0)

        for face in faces:
            landmarks = self.predictor(img_gray, face)

            point_30_x = landmarks.part(30).x
            point_30_y = landmarks.part(30).y

            point_36_x = landmarks.part(36).x
            point_36_y = landmarks.part(36).y

            point_39_x = landmarks.part(39).x
            point_39_y = landmarks.part(39).y

            point_42_x = landmarks.part(42).x
            point_42_y = landmarks.part(42).y

            point_45_x = landmarks.part(45).x
            point_45_y = landmarks.part(45).y

            point_48_x = landmarks.part(48).x
            point_48_y = landmarks.part(48).y

            point_54_x = landmarks.part(54).x
            point_54_y = landmarks.part(54).y

            cv.circle(frame, (point_30_x, point_30_y), 2, (0,255,0), 3)
            cv.circle(frame, (point_36_x, point_36_y), 2, (0,255,0), 3)
            cv.circle(frame, (point_39_x, point_39_y), 2, (0,255,0), 3)
            cv.circle(frame, (point_42_x, point_42_y), 2, (0,255,0), 3)
            cv.circle(frame, (point_45_x, point_45_y), 2, (0,255,0), 3)
            cv.circle(frame, (point_48_x, point_48_y), 2, (0,255,0), 3)
            cv.circle(frame, (point_54_x, point_54_y), 2, (0,255,0), 3)
        
        return frame


    def draw_line(self,frame):
        img_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces = self.detector(img_gray, 0)

        for face in faces:
            landmarks = self.predictor(img_gray, face)
            # Center Point
            point_30_x = landmarks.part(30).x
            point_30_y = landmarks.part(30).y

            point_36_x = landmarks.part(36).x
            point_36_y = landmarks.part(36).y

            point_39_x = landmarks.part(39).x
            point_39_y = landmarks.part(39).y
            

            point_42_x = landmarks.part(42).x
            point_42_y = landmarks.part(42).y

            point_45_x = landmarks.part(45).x
            point_45_y = landmarks.part(45).y

            point_48_x = landmarks.part(48).x
            point_48_y = landmarks.part(48).y

            point_54_x = landmarks.part(54).x
            point_54_y = landmarks.part(54).y

            cv.line(frame, (point_36_x, point_36_y), (point_30_x, point_30_y), (0,0,255), 1)
            cv.line(frame, (point_39_x, point_39_y), (point_30_x, point_30_y), (0,0,255), 1)
            cv.line(frame, (point_42_x, point_42_y), (point_30_x, point_30_y), (0,0,255), 1)
            cv.line(frame, (point_45_x, point_45_y), (point_30_x, point_30_y), (0,0,255), 1)
            cv.line(frame, (point_48_x, point_48_y), (point_30_x, point_30_y), (0,0,255), 1)
            cv.line(frame, (point_54_x, point_54_y), (point_30_x, point_30_y), (0,0,255), 1)

            point_center = []
            point_center.append(point_30_x)
            point_center.append(point_30_y)
            point_center = np.array(point_center)


            # Feature 1
            point_1 = []
            point_1.append(point_36_x)
            point_1.append(point_36_y)
            distance = self.euclidean_distance(point_center, point_1)

            point_2 = []
            point_2.append(point_39_x)
            point_2.append(point_39_y)
            distance = self.euclidean_distance(point_center, point_2)

            point_3 = []
            point_3.append(point_42_x)
            point_3.append(point_42_y)
            distance = self.euclidean_distance(point_center, point_3)

            point_4 = []
            point_4.append(point_45_x)
            point_4.append(point_45_y)
            distance = self.euclidean_distance(point_center, point_4)

            point_5 = []
            point_5.append(point_48_x)
            point_5.append(point_48_x)
            distance = self.euclidean_distance(point_center, point_5)

            point_6 = []
            point_6.append(point_54_x)
            point_6.append(point_54_x)
            distance = self.euclidean_distance(point_center, point_6)

            return frame, self.features

    def euclidean_distance(self, point_x, point_y):
        different = point_x - point_y
        different_sum = np.sum(different ** 2)
        distance = np.sqrt(different_sum)
        self.features.append(distance)

        return distance

    def knn_predict(self, features, frame):
        feature_pred = np.array(features)
        feature_pred = feature_pred.reshape(1, -1)
        model_time = time.time()
        y_pred = self.model.predict(feature_pred)
        model_time = time.time() - model_time
        frame = cv.putText(frame, self.person[y_pred[0]], (30,60), cv.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0),2)
        print(f'Waktu Komputasi Model : {model_time}')
        return frame
    



    
    
