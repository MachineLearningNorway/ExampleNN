import sys
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from keras.datasets import mnist
from keras.models import model_from_json

is_CNN_mode = True

if is_CNN_mode:
    progname = "MNIST demo CNN"
else:
    progname = "MNIST demo Perceptron"

(X_train, y_train), (X_test, y_test) = mnist.load_data()


if is_CNN_mode:
    X_test_reshaped = X_test.reshape(X_test.shape[0], 28, 28, 1)
else:
    X_test_reshaped = X_test.reshape(10000, 784)

# load json and create model
if is_CNN_mode:
    json_file = open('mnist.cnn.json', 'r')
else:
    json_file = open('mnist.perceptron.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
if is_CNN_mode:
    loaded_model.load_weights("mnist.cnn.h5")
else:
    loaded_model.load_weights("mnist.perceptron.h5")
print("Loaded model from disk")

index = np.random.randint(1, high=10000)

def predict_digit():
    global index
    y_ = loaded_model.predict(X_test_reshaped[index:index + 1], verbose=1)
    s = np.sum(y_[0] * [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(int)
    return s

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)

    def draw_picture_28x28(self):
        global index
        index = np.random.randint(1, high=10000)
        pixels = X_test[index].astype('uint8')
        self.axes.imshow(pixels, cmap='gray')

    def compute_initial_figure(self):
        self.draw_picture_28x28()

    def update_figure(self):
        self.draw_picture_28x28()
        self.draw()


class MyDynamicQLabel(QtWidgets.QLabel):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        QtWidgets.QLabel.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)


class MyDynamicMNISTLabel(MyDynamicQLabel):
    def __init__(self, *args, **kwargs):
        MyDynamicQLabel.__init__(self, *args, **kwargs)

    def compute_initial_figure(self):
       global index
       self.setText(str(y_test[index]))

    def update_figure(self):
        self.setText(str(y_test[index]))

class MyDynamicMNISTPredicted(MyDynamicQLabel):
    def __init__(self, *args, **kwargs):
        MyDynamicQLabel.__init__(self, *args, **kwargs)

    def compute_initial_figure(self):
        s = predict_digit()
        self.setText(str(s))

    def update_figure(self):
        global index
        s = predict_digit()
        self.setText(str(s))

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtWidgets.QWidget(self)

        l = QtWidgets.QVBoxLayout(self.main_widget)

        dc = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)

        l1 = QtWidgets.QHBoxLayout(self.main_widget)

        la = QtWidgets.QLabel("Label")
        la.setFont(QtGui.QFont("Times", 28, QtGui.QFont.Bold))

        la1 = MyDynamicMNISTLabel("")
        la1.setFont(QtGui.QFont("Times", 28, QtGui.QFont.Bold))
        l1.addWidget(la)
        l1.addWidget(la1)

        l2 = QtWidgets.QHBoxLayout(self.main_widget)

        lp = QtWidgets.QLabel("Predicted")
        lp.setFont(QtGui.QFont("Times", 28, QtGui.QFont.Bold))

        lp1 = MyDynamicMNISTPredicted("")
        lp1.setFont(QtGui.QFont("Times", 28, QtGui.QFont.Bold))

        l2.addWidget(lp)
        l2.addWidget(lp1)

        l.addWidget(dc)
        l.addLayout(l1)

        l.addLayout(l2)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "MNIST demo",
                                    """MNIST demo"""
                                )

qApp = QtWidgets.QApplication(sys.argv)

aw = ApplicationWindow()
aw.setWindowTitle("%s" % progname)
aw.show()
sys.exit(qApp.exec_())
#qApp.exec_()