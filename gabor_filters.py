import numpy as np
import cv2
import dlib


def build_filters():
    """
    Returns a list of 8 Gabor filters with theta between 0 and pi
    
    Generate 8 Gabor kernels based on preset parameters and varying orientations (theta).
    Parameters:
        kernel size = 31x31
        sigma = 4.4
        lambda = 10.0
        gamma = 0.5
        psi = 0
    
    Returns:
        [cv2.CV_64F]: list of filters created with OpenCV
    """
    
    filters = []
    ksize = 31
    #sig = 4.5
    lam = 10.0
    gam = 0.5
    psi = 0

    sig = 4.4

    for theta in np.arange(0, np.pi, np.pi/8):
        kern = cv2.getGaborKernel((ksize,ksize), sig, theta, lam, gam, psi, ktype=cv2.CV_64F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    
    return filters
    
if __name__ == '__main__':
    """
    Test function to apply Gabor filters to input image and display comparison
    
    Args:
        sys.argv[1] (str): file directory to input image
        
    """
    
    import sys
    
    img_fn = sys.argv[1]
    
    img = cv2.imread(img_fn)
    #img = cv2.resize(img, (300, 200))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    filters = build_filters()
    
    
    cv2.imshow('initial image', img)
    cv2.waitKey(0)
    
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    face = detector(img, 1)[0]
    sf = 100.0 / float(face.right()-face.left())
    #print(sf)
    # img = img[face.top():face.bottom(), face.left():face.right()]
    img = cv2.resize(img, None, fx=sf, fy=sf)
    
    for filter in filters:
        cv2.imshow('filter', filter)
        img_filtered = cv2.filter2D(img, cv2.CV_8UC1, filter)
        cv2.imshow('filtered image', img_filtered)
        #print(img_filtered[100][100])
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()