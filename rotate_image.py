import sys
import cv2
import numpy as np
import dlib

predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

## Points used to line up the images.    
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
    RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

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
    
def draw_points(img, points):
    """
    Draws list of points on input image
    
    Args:
        img (numpy.ndarray): input image
        points ((int, int)): list of pairs of x and y coordinates corresponding to fiducial points
    
    """
    
    for p in points:
        cv2.circle(img, p, 2, color=(0,0,255))
        
def draw_points_numbered(img, points):
    """
    Draws numbered list of points on input image
    
    Args:
        img (numpy.ndarray): input image
        points ((int, int)): list of pairs of x and y coordinates corresponding to fiducial points
    
    """
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for n, p in enumerate(points):
        cv2.putText(img, str(n), p, font, 0.25, (0,0,255))
        
def draw_face(img, face):
    """
    Draws bounding box of face on input image
    
    Args:
        img (numpy.ndarray): input image
        face (dlib.rectangle): bounding rectangle of a face within input image
    
    """
    
    cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)


def transformation_from_points(points1, points2):
    """
    Align points in first image so they fit as close as possible to points in second image
    
    Create an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    
    Args:
        points1 ((int, int)): list of pairs of x and y coordinates corresponding to fiducial points
        points2 ((int, int)): list of pairs of x and y coordinates corresponding to fiducial points
        
    Returns:
        numpy.ndarray: Affine transformation matrix
    
    """

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])
                        
def warp_im(im, M, dshape):
    """
    Map input image onto base image 
    
    Args:
        im (numpy.ndarray): input image
        M (numpy.ndarray): Affine transformation matrix
        dshape ([int]): shape of base image
    
    """
    
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im        

def cut_face(img, face):
    """
    Extract face from input image
    
    Args:
        img (numpy.ndarray): input image
        face (dlib.rectangle): bounding rectangle of a face within input image
        
    Returns:
        numpy.ndarray: new image with face extracted and padding added
        
    """

    ## Add padding for rotation
    add_size = int(face.width()*0.2)
    img2 = img[face.top()-add_size:face.bottom()+add_size, face.left()-add_size:face.right()+add_size]

    ## Resize face to 100x100
    sf = 100.0 / float(face.right()-face.left())
    img2 = cv2.resize(img2, None, fx=sf, fy=sf)
    
    return img2
    
def rotate_image(img):
    """
    Rotates extracted face to a neutral orientation
    
    Args:
        img (numpy.ndarray): input image
        
    Returns:
        numpy.ndarray: output image with rotated face
        
    """

    ## Get size of resized image
    dim = img.shape[0]

    ## Create rectangle size of resized image (to place blank face)

    face2 = dlib.rectangle(0, 0, dim, dim)

    ## Input image points

    img_face_shape = predictor(img, face2)
    img_points = np.matrix([[p.x, p.y] for p in img_face_shape.parts()])

    ## Empty image to base points

    empty_img = np.empty([dim,dim], dtype=np.uint8)
    empty_img.fill(255)

    empty_face_shape = predictor(empty_img,face2)
    base_points = np.matrix([[p.x, p.y] for p in empty_face_shape.parts()])

    ## Perform transformation
    M = transformation_from_points(base_points[ALIGN_POINTS], img_points[ALIGN_POINTS])
    warped_img = warp_im(img, M, img.shape)
    
    return warped_img

if __name__ == '__main__':
    """
    Test function to rotate input image and display result
    
    Args:
        sys.argv[1] (str): file directory to input image
        
    """
    
    ## Read in image
    img_fn = sys.argv[1]

    img = cv2.imread(img_fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ## Locate face
    face = get_face(img)
    
    ## Cut face
    img2 = cut_face(img, face)
    
    ## Rotate image
    warped_img = rotate_image(img2)
    
    ## Get new face points
    dim = img2.shape[0]
    face2 = dlib.rectangle(0, 0, dim, dim)
    warped_img_face_shape = predictor(warped_img, face2)
    img_points2 = [(p.x, p.y) for p in warped_img_face_shape.parts()]
    draw_points(warped_img, img_points2)
    
    cv2.imshow('rotated image', warped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




