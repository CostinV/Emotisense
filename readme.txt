Training the Neural Network:

The python script train_nn_classifier reads in image files from subdirectory “facegrabber_new”, calculates a feature vector for each one, and trains a neural network classifier to recognize various emotions - neutral, angry, surprised, happy, and sad - based on those values, saving its output to nnclf2.pk1 for later use.
If the trained file is already provided, this script does not need to be run.

Usage:
python train_nn_classifier.py

Training the Facial Recognition:
       In order to get a good sample size for recognition, a script was put together to allow a user to sit in front of a computer, and run the webcam to take many images of them within a small time from of 5-10 seconds. This was then fed into the recognizer object, and saved the training into a file called recognizer_training. As in the neural network, if this file is provided, you don’t need to retrain- but, the file can be updated to add new images to the training in order for it to work for new people.

If a folder with training files is preferred to this method, there is a different training file available to use called: training_recognizer.py. 

Usage:
python  train_recognizer_auto.py
OR (the secondary way)
python training_recognizer.py 

Running the software:
	To run the software the statement ‘python emotisense_gui.py’ should be inputted into the command line. A screen will open with three options ‘Video Feed’, ‘Video File’, and ‘Quit’. Quit will close the window and program. The Video Feed option will start the webcam and begin taking in frames while closing the start window and opening a new window displaying these frames, a table of the number of each emotion within the frame, and a graph displaying emotions versus frame number. An emotion label will also be displayed above each detected face. Choosing Video File will display a dialog box asking the user to choose a video file to display. Upon choosing a file, a similar window as described above will open, playing the video and displaying the same functionalities as in with the Video Feed option.   	

Usage:
python emotisense_main.py 



Library Dependencies:

PyQT 4
OpenCV (2.4.X, lower than 3.X for facial recognition)
Dlib
Numpy
Scikit-Learn
Scikit-NeuralNetwork
Matplotlib
Pickle
Glob

