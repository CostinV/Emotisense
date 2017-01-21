import sys
from PyQt4 import QtGui, QtCore
import cv2
import video_feed
import video_file


class ChooseFeedWindow(QtGui.QMainWindow):
    """
    First application window
    
    Allow user to choose between video file and video feed for analysis.
    Switch to selected window once button is pressed.
    """

	def __init__(self):
		super(ChooseFeedWindow, self).__init__()  # returns main object
		self.setGeometry(50, 50, 200, 200)
		self.setWindowTitle('Emotisense')
		# self.setWindowIcon(QtGui.QIcon(''))

		label = QtGui.QLabel('Emotisense', self)
		label.move(60, 10)

		font = QtGui.QFont()
		font.setBold(True)
		label.setFont(font)

		button_video_feed = QtGui.QPushButton('Video Feed', self)
		button_video_feed.clicked.connect(self.video_feed)
		button_video_feed.setFixedWidth(100)
		button_video_feed.move(50, 50)

		button_video_file = QtGui.QPushButton('Video File', self)
		button_video_file.clicked.connect(self.video_file)
		button_video_file.setFixedWidth(100)
		button_video_file.move(50, 85)

		button_quit = QtGui.QPushButton('Quit', self)
		button_quit.setFixedWidth(100)
		button_quit.clicked.connect(self.close_app)
		button_quit.move(50, 135)

	def close_app(self):
		sys.exit()

	def video_file(self):
		global video_file_gui
		video_file_gui = video_file.VideoFileWindow()
		video_file_gui.show()
		gui.close()

	def video_feed(self):
		global video_feed_gui
		video_feed_gui = video_feed.VideoFeedWindow()
		video_feed_gui.show()
		gui.close()




app = QtGui.QApplication(sys.argv)

gui = ChooseFeedWindow()
gui.show()
sys.exit(app.exec_())
run()


