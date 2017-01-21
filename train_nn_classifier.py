from sknn.mlp import Regressor, Layer, Classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import glob
import gabor_filters
import parse_jaffe_data
import re
import dlib
import cv2
import numpy as np
import sys
import os
import logging
import pickle
import rotate_image

"""
This module trains and saves a neural network classifier based on selected input images
and manually created output labels

"""

# enable logging
logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG,
    stream=sys.stdout)

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
    #points = points[17:]
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
    
def stringSplitByNumbers(x):
    """Creates a key to sort by appended number rather than characters"""

    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

training_files = glob.glob('FaceGrabberNew/*')
training_files = sorted(training_files, key = stringSplitByNumbers)

training_input = []

# read in and process training files from directory
for num, file in enumerate(training_files):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face = get_face(img)
    face_area = rotate_image.cut_face(img, face)
    face_area = rotate_image.rotate_image(face_area)
    dim = face_area.shape[0]
    face2 = dlib.rectangle(0, 0, dim, dim)

    training_input.append(generate_feature_vector(face_area, face2))
    print("Processed training file ", num)

train_x = np.array(training_input)

# manually created labels to train on
training_output = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

train_y = np.array(training_output)
 
# create pipeline to scale input features between 0.0 and 1.0 and pass to neural network classifier
pipeline = Pipeline([
    ('min/max scaler', MinMaxScaler(feature_range=(0.0,1.0))),
    ('neural network', Classifier(
        layers = [
            Layer("Sigmoid", units=400),
            Layer("Softmax")],
        learning_rate = 0.01,
        n_iter = 300))])

# train neural network with specified parameters 
pipeline.fit(train_x, train_y)

# save trained network to disk
pickle.dump(pipeline, open('nnclf2.pk1', 'wb'))

print("done")