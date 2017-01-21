from PyQt4 import QtGui, QtCore
import capture
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


class VideoFeedWindow(QtGui.QWidget):
    """
    User interface window for video feed

    Integrate capture window, display table and graph based on emotion results
        
    """
    
    def pause(self):
        self.video.stop()

    def play(self):
        self.video.start()

    def __init__(self, parent=None):

        super(VideoFeedWindow, self).__init__(parent)
        
        self.graph_x = [0]
        self.frame_number = 0
        self.graph_y_neutral = [0]
        self.graph_y_angry_disgust = [0]
        self.graph_y_fear_surprise = [0]
        self.graph_y_happy = [0]
        self.graph_y_sad = [0]
 
        
        fig = plt.figure(figsize=(5,5))
        ax1 = fig.add_subplot(111)
        ax1.plot(self.graph_x, self.graph_y_neutral)
        fig.canvas.draw()

        w,h = fig.canvas.get_width_height()
        graph_np = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph_np.shape = (w, h, 3)
        
        image = QtGui.QImage(graph_np, w, h, QtGui.QImage.Format_RGB888)
        image = QtGui.QPixmap.fromImage(image)
        self.graph_img = image
        
        self.set_up()

    def set_up(self):
        """
        Initialization function for video feed window
        
        Create capture window.
        Set frame, button, table, and graph labels.
        Create and set layout.
        
        """
        
        self.setGeometry(50, 50, 800, 800)
        self.video = capture.Capture(cv2.VideoCapture(0), self)
        self.video.start()

        font = QtGui.QFont('Times', 16, QtGui.QFont.Bold)

        video_feed_label = QtGui.QLabel('Video Feed', self)
        video_feed_label.setFont(font)
        video_feed_label.setAlignment(QtCore.Qt.AlignCenter)

        play_button = QtGui.QPushButton('Resume', self)
        play_button.clicked.connect(self.play)

        pause_button = QtGui.QPushButton('Pause', self)
        pause_button.clicked.connect(self.pause)

        main_layout = QtGui.QVBoxLayout()
        main_layout.setAlignment(QtCore.Qt.AlignTop)

        button_layout = QtGui.QHBoxLayout()
        button_layout.addWidget(play_button)
        button_layout.addWidget(pause_button)
        
        emotion_table = QtGui.QTableWidget()
        self.emotion_table = emotion_table
        emotion_table.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)

        emotion_table.setHorizontalHeaderLabels(QtCore.QString('Emotion;Count').split(';'))
        emotion_table.setRowCount(5)
        emotion_table.setColumnCount(2)
        
        emotion_table.setItem(0, 0, QtGui.QTableWidgetItem('Neutral'))
        emotion_table.setItem(1, 0, QtGui.QTableWidgetItem('Angry/Disgust'))
        emotion_table.setItem(2, 0, QtGui.QTableWidgetItem('Fear/Surprise'))
        emotion_table.setItem(3, 0, QtGui.QTableWidgetItem('Happy'))
        emotion_table.setItem(4, 0, QtGui.QTableWidgetItem('Sad'))
        
        graph_label = QtGui.QLabel()
        self.graph_label = graph_label
        graph_label.setPixmap(self.graph_img)

        graph_layout = QtGui.QHBoxLayout()
        graph_layout.addWidget(emotion_table)
        
        
        graph_layout.addWidget(graph_label)
        

        main_layout.addWidget(video_feed_label)
        main_layout.addWidget(self.video)
        main_layout.addLayout(graph_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        
    def update_graph(self,emotions):
        """
        Callback function to update graph when emotions are computed for current frame
        
        Args:
            emotions ([int]): array mapping enumerated emotions to count in current frame
            
        """
    
        ## graph
        #print "updating graph"
        self.frame_number += 1
        self.graph_x.append(self.frame_number)
        self.graph_y_neutral.append(emotions[0])
        self.graph_y_angry_disgust.append(emotions[1])
        self.graph_y_fear_surprise.append(emotions[2])
        self.graph_y_happy.append(emotions[3])
        self.graph_y_sad.append(emotions[4])
        
        fig = plt.figure(figsize=(5,5))
        newplot = fig.add_subplot(111)
        newplot.plot(self.graph_x[-20:], self.graph_y_neutral[-20:], label = "Neutral")
        newplot.plot(self.graph_x[-20:], self.graph_y_angry_disgust[-20:], label = "Anger/Disgust")
        newplot.plot(self.graph_x[-20:], self.graph_y_fear_surprise[-20:], label = "Fear/Surprise")
        newplot.plot(self.graph_x[-20:], self.graph_y_happy[-20:], label = "Happy")
        newplot.plot(self.graph_x[-20:], self.graph_y_sad[-20:], label = "Sad")
        newplot.set_ylim((0,5))
        newplot.set_xlabel("Frame No")
        newplot.set_ylabel("Number of Emotions")
        newplot.legend()
        fig.canvas.draw()
        
        w,h = fig.canvas.get_width_height()
        graph_np = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph_np.shape = (w, h, 3)

        image = QtGui.QImage(graph_np, w, h, QtGui.QImage.Format_RGB888)
        image = QtGui.QPixmap.fromImage(image)
        self.graph_label.setPixmap(image)
        
        #print(self.graph_x)
        #print(self.graph_y_neutral)

    def update_table(self, emotions):
        """
        Callback function to update table when emotions are computed for current frame
        
        Args:
            emotions ([int]): array mapping enumerated emotions to count in current frame
        
        """
        print 'Emotion Count: ' + str(emotions)
        
        self.emotion_table.setItem(0, 1, QtGui.QTableWidgetItem(str(emotions[0])))
        self.emotion_table.setItem(1, 1, QtGui.QTableWidgetItem(str(emotions[1])))
        self.emotion_table.setItem(2, 1, QtGui.QTableWidgetItem(str(emotions[2])))
        self.emotion_table.setItem(3, 1, QtGui.QTableWidgetItem(str(emotions[3])))
        self.emotion_table.setItem(4, 1, QtGui.QTableWidgetItem(str(emotions[4])))
        