from sknn.mlp import Regressor, Layer, Classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import glob
import gabor_filters
import parse_jaffe_data
import dlib
import cv2
import numpy as np
import sys
import logging
import pickle
import rotate_image

# enable logging
logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG,
    stream=sys.stdout)

pipeline = pickle.load(open('nnclf2.pk1', 'rb'))


predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

gf = gabor_filters.build_filters()

def generate_feature_vector(img, face):
    """
    Generates feature vector from input image and face coordinates
    
    8 Gabor filter applied at 48 fiducial points each generates a feature vector of size 384
    
    Args:
        img (numpy.ndarray): input image
        face (dlib.rectangle): bounding rectangle of a face within input image
        
    Returns:
        [int]: list of 384 features generated from input image and face coordinates
        
    """
    feature_vector = []
    shape = predictor(img, face)
    points = [(p.x, p.y) for p in shape.parts()]

    points = points[17:]
    points = points[0:10] + points[14:]

    for filter in gf:
        img_filtered = cv2.filter2D(img, cv2.CV_8UC1, filter)
        for (x,y) in points:
            feature_vector.append(img_filtered.item(y,x))
            
    return feature_vector

def get_face(img):
    """
    Retrieve first detected face from input image
    
    Args:
        img (numpy.ndarray): input image
        
    Returns:
        dlib.rectangle: bounding rectangle of a face within input image
        
    """
    
    face = detector(img, 1)[0]
    return face
    
def get_faces(img):
    """
    Retrieve all detected faces from input image
    
    Args:
        img (numpy.ndarray): input image
        
    Returns:
        [dlib.rectangle]: list of bounding rectangles of faces within input image
        
    """
    
    faces = detector(img, 1)
    return faces

def get_emotions(img):
    """
    Compute emotions of faces within input image
    
    Args:
        img (numpy.ndarray): input image
        
    Returns:
        [int]: list of enumerated emotions corresponding to faces within input image
    """
    
    emotions = {}
    emotions[0] = [0]
    emotions[1] = [0]
    emotions[2] = [0]
    emotions[3] = [0]
    emotions[4] = [0]
    
    faces = get_faces(img)
    
    if faces is None:
        return emotions
    
    for face in faces:
        face_area = rotate_image.cut(image, face)
        face_area = rotate_image.rotate_face(face_area)
        dim = face_area.shape[0]
        face2 = dlib.rectangle(0, 0, dim, dim)
        face_vector = generate_feature_vector(face_area, face2)
        test_x = np.array([test_feature_vector])
        test_y = pipeline.predict(test_x)

        emotions[test_y] = 1
    
    return emotions
    

if __name__ == '__main__':  
    """
    Test function to compute emotions in input image and display result
    
    Args:
        sys.argv[1] (str): file directory to input image
        
    """  

    img_fn = sys.argv[1]

    test_img = cv2.imread(img_fn)

    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    
    test_face = get_face(test_img)
    
    face_area = rotate_image.cut_face(test_img, test_face)
    face_area = rotate_image.rotate_image(face_area)
    dim = face_area.shape[0]
    face2 = dlib.rectangle(0, 0, dim, dim)
    

    test_feature_vector = generate_feature_vector(face_area, face2)
    test_x = np.array([test_feature_vector])

    test_y = pipeline.predict(test_x)
    
    print(test_y)
