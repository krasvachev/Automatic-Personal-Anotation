from enum import Enum
import os
import sys
import re
import numpy as np
from PIL import Image 
import cv2 as cv2

class Dataset(Enum):
    YALE_EX_CROPPED = 'CroppedYale'
    YALE_EX = 'ExtendedYaleB'
    TU = 'TU'

class DataLoader:

    def __init__(self, dataset_type, settings):
        if(dataset_type not in [item for item in Dataset]):
            raise Exception("Nonexistant dataset!")

        self.__dataset_name = str.lower(dataset_type.name)
        self.__dataset_type = dataset_type
        self.__dir = 'datasets/' + dataset_type.value
        self.__opt_duration = None
        self.__feature_length = None
        self.__settings = settings

    def __crop_image(self, img, sizeCheck = False):
        
        face_rects = self.__face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

        if(len(face_rects) == 0):
            return False, 0

        x, y, w, h = face_rects[0]
        crop_img = img[y:y+h, x:x+w]
        if(w < 125 or h < 125):
            return False, 0

        _, landmarks = self.__landmark_detector.fit(img, face_rects)

        if(len(landmarks) == 0):
            return False, 0

        shape = landmarks[0][0]
        boundRect = cv2.boundingRect(shape)
        brX1, brY1, brW, brH = boundRect

        out_face = np.zeros_like(img)

        remapped_shape = np.zeros_like(shape) 
        feature_mask = np.zeros((img.shape[0], img.shape[1]))
        remapped_image = cv2.convexHull(shape).astype('int32') 

        cv2.fillConvexPoly(feature_mask, remapped_image, 1)
        feature_mask = feature_mask.astype(np.bool)
        out_face[feature_mask] = img[feature_mask]
        out_face = out_face[brY1 : brY1 + brH, brX1 : brX1 + brW]

        return True, out_face

    def __resize_image(self, img, new_size):
        pass

    def load_data(self, reload=False):

        settings_str  = '_'.join([str(obj[0]) + "-" + str(obj[1]) for obj in self.__settings.items()])
        ready_file_path =  "datasets\\ready\\" + self.__dataset_name + "_" + settings_str + ".npy"

        if(os.path.isfile(ready_file_path) and reload==False):
            with open(ready_file_path, 'rb') as f:
                rdy_arr = np.load(f, allow_pickle=True)
                X, y, z, v = rdy_arr[0], rdy_arr[1], rdy_arr[2], rdy_arr[3]

                return X,y,z,v

        X = None
        flag = 0
        y = np.array([], dtype="uint8")
        v = None



        if(self.__dataset_type == Dataset.YALE_EX_CROPPED or self.__dataset_type == Dataset.YALE_EX):
            self.__face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            self.__landmark_detector = cv2.face.createFacemarkLBF()
            self.__landmark_detector.loadModel('lbfmodel.yaml')

            for root, subFolders, files in os.walk(self.__dir, topdown=True):
                for file in files:
                    if(file.endswith('.pgm')):
                        pattern = re.compile(r"yaleB(\d+)+_P(\d+)+A([-+]\d+)+E([-+]\d+)+.pgm")
                        result = pattern.match(file)
                        if(result == None):
                            continue
                        groups = result.groups()
                        if(len(groups) != 4):
                            continue

                        person_n = int(groups[0])
                        #face_angle = int(groups[1])
                        light_azi = int(groups[2])
                        light_elev = int(groups[3])

                        im = Image.open(root + '/' + file)
                        numIm = np.array(im)
                        resIm = numIm
                        #resIm = cv2.equalizeHist(numIm)

                        if self.__dataset_type == Dataset.YALE_EX:
                            resIm = self.__crop_image(resIm, True)
                            if(resIm[0] == False):
                                continue
                            resIm = cv2.resize(resIm[1], (100,100), interpolation = cv2.INTER_CUBIC)

                        if flag == 0:
                            X = resIm
                            flag = 1
                        else:
                            X = np.dstack((X, resIm))
                        
                        y = np.append(y, person_n)
        
            X = np.transpose(X,(2,0,1))

        elif(self.__dataset_type == Dataset.TU ):
            for root, subFolders, files in os.walk(self.__dir, topdown=True):
                for file in files:
                    pattern = re.compile(r"(\d+)%([-]*\d+)%([-]*\d+)%([-]*\d+).jpg")
                    result = pattern.match(file)
                    if(result == None):
                        continue
                    groups = result.groups()
                    if(len(groups) != 4):
                        continue

                    person_n = int(groups[0])
                    angle_y = int(groups[1])
                    angle_p = int(groups[2])
                    angle_r = int(groups[3])

                    if('angle_limit' in self.__settings):
                        ang = self.__settings['angle_limit']
                        if(angle_y > ang or angle_p > ang & angle_r > ang):
                            continue

                    im_cur = cv2.imread(root + '/' + file, self.__settings['img_format'])
                    resIm = cv2.resize(im_cur, (75,100), interpolation = cv2.INTER_CUBIC)
                    if(len(resIm.shape) == 3):
                        resIm = np.reshape(resIm, (1, resIm.shape[0],resIm.shape[1],resIm.shape[2]))
                        #print("Shape of resIm")
                        #print(resIm.shape)

                    if flag == 0:
                        X = resIm
                        flag = 1
                    else:
                        X = np.concatenate((X, resIm), axis=0)
                    
                    y = np.append(y, person_n)
     
        v = np.unique(y)
        z = len(v)

        indices = list(range(0,z))
        dict_tmp = dict(zip(v.tolist(),indices))
        y = np.array([dict_tmp.get(i, -1) for i in y])

        with open(ready_file_path, 'wb') as f:
            np.save(f, np.array([X, y, z, v]))

        return X,y,z,v



