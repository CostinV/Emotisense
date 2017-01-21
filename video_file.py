__author__ = 'Wendy Kopf'

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QScrollBar
import cv2
import capture
import pyqtgraph

class VideoFileWindow(QtGui.QWidget):
    """
    User interface window for video file

    Integrate capture window, display table and graph based on emotion results
        
    """

    def __init__(self, parent=None):

        super(VideoFileWindow, self).__init__(parent)
        self.setGeometry(50, 50, 300, 300)
        path = QtGui.QFileDialog.getOpenFileName()
        self.video = capture.Capture(cv2.VideoCapture(unicode(path)), self, self.next_frame)
        self.video.start()
        self.video.show()


        font = QtGui.QFont('Times', 16, QtGui.QFont.Bold)
        
        video_file_label = QtGui.QLabel('Video File', self)
            video_file_label.setFont(font)
            video_file_label.setAlignment(QtCore.Qt.AlignCenter)

            pause_button = QtGui.QPushButton('Pause', self)
            pause_button.clicked.connect(self.pause)
        play_button = QtGui.QPushButton('Play', self)
            play_button.clicked.connect(self.play)
        
        results_button = QtGui.QPushButton('View Results', self)
            # results_button.clicked.connect(self.results)
        
        button_layout = QtGui.QHBoxLayout()
            button_layout.addWidget(pause_button)
        button_layout.addWidget(play_button)
        button_layout.addWidget(results_button)
            
            emotion_table = QtGui.QTableWidget()
            emotion_table.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)

            emotion_table.setHorizontalHeaderLabels(QtCore.QString('Emotion;Count').split(';'))
            emotion_table.setRowCount(6)
            emotion_table.setColumnCount(2)
            emotion_table.setItem(0, 0, QtGui.QTableWidgetItem('Happy'))
            emotion_table.setItem(1, 0, QtGui.QTableWidgetItem('Sad'))
            emotion_table.setItem(2, 0, QtGui.QTableWidgetItem('Angry'))
            emotion_table.setItem(3, 0, QtGui.QTableWidgetItem('Inattentive'))
            emotion_table.setItem(4, 0, QtGui.QTableWidgetItem('Disgust'))
        emotion_table.setItem(5, 0, QtGui.QTableWidgetItem('Neutral'))
        
        plot = pyqtgraph.PlotWidget()
            plot.setXRange(0, 20, padding=.001)
            plot.setYRange(0, 20, padding=.001)
        
        self.scrollbar = QtGui.QScrollBar(QtCore.Qt.Horizontal)
        self.scrollbar.setRange(0, self.video.video_source.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self.scrollbar.sliderMoved.connect(self.scrolled)
            
            main_layout = QtGui.QVBoxLayout()
            main_layout.setAlignment(QtCore.Qt.AlignTop)
        
        graph_layout = QtGui.QHBoxLayout()
            graph_layout.addWidget(emotion_table)
            graph_layout.addWidget(plot)
        
            main_layout.addWidget(video_file_label)
            main_layout.addWidget(self.video)
            main_layout.addLayout(graph_layout)
            main_layout.addLayout(button_layout)
        main_layout.addWidget(self.scrollbar)
     

        self.setLayout(main_layout)
    def next_frame(self):
        self.scrollbar.setSliderPosition(self.scrollbar.sliderPosition() + 1)
    def pause(self):
         self.video.stop()
    def play(self):
         self.video.start()
    def scrolled(self, new_value):
         self.video.scroll(new_value)
	                
