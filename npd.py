import sys
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QImage,
    QPixmap
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QDesktopWidget,
    QHBoxLayout,
    QWidget,
    QScrollArea,
    QFileDialog,
    QAction,
    QLabel,
    QSpinBox,
    QStyle
)


class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Nanoparticle Detector')
        geometry = self.frameGeometry()
        geometry.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(geometry.topLeft())
        self.image_a, self.image_b = None, None
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        scroll_area = QScrollArea()
        scroll_area.setAlignment(Qt.AlignCenter)
        scroll_area.setWidget(image_label)
        self.setCentralWidget(scroll_area)
        open_icon = self.style().standardIcon(QStyle.SP_DialogOpenButton)
        save_icon = self.style().standardIcon(QStyle.SP_DialogSaveButton)
        self.action_open = QAction(open_icon, 'Open', self)
        self.action_open.setShortcut('Ctrl+O')
        self.action_open.triggered.connect(self.open)
        self.action_save = QAction(save_icon, 'Save', self)
        self.action_save.setShortcut('Ctrl+S')
        self.action_save.triggered.connect(self.save)
        self.action_save.setEnabled(False)
        toolbar = self.addToolBar('File')
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.addAction(self.action_open)
        toolbar.addAction(self.action_save)
        toolbar.addSeparator()
        self.kernel_blur = QSpinBox()
        self.kernel_blur.setValue(3)
        self.kernel_blur.setRange(1, 100)
        self.kernel_tophat = QSpinBox()
        self.kernel_tophat.setValue(11)
        self.kernel_tophat.setRange(1, 100)
        self.threshold = QSpinBox()
        self.threshold.setValue(50)
        self.threshold.setRange(1, 255)
        for widget, name in ((self.kernel_blur, 'Blur Kernel Size'),
                             (self.kernel_tophat, 'Top-Hat Kernel Size'),
                             (self.threshold, 'Threshold')):
            widget.setEnabled(False)
            widget.valueChanged[int].connect(self.refresh)
            h = QHBoxLayout()
            h.addWidget(QLabel(name))
            h.addWidget(widget)
            w = QWidget()
            w.setLayout(h)
            toolbar.addWidget(w)

    def open(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open')
        if fname is not None:
            self.image_a = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            for widget in (self.kernel_blur,
                           self.kernel_tophat,
                           self.threshold):
                widget.setEnabled(True)
            self.refresh()

    def save(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save', 'output.png')
        if fname is not None:
            cv2.imwrite(fname, self.image_b)

    def refresh(self):
        mask = self.image_a
        mask = cv2.blur(
            mask,
            (self.kernel_blur.value(),
             self.kernel_blur.value())
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.kernel_tophat.value(),
                 self.kernel_tophat.value()),
            )
        )
        _, mask = cv2.threshold(
            mask,
            self.threshold.value(),
            255,
            cv2.THRESH_BINARY
        )
        _, contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        self.image_b = cv2.drawContours(
            cv2.cvtColor(self.image_a, cv2.COLOR_GRAY2RGB),
            contours,
            -1,
            (0, 255, 0),
            2,
        )
        self.action_save.setEnabled(True)
        self.centralWidget().widget().setPixmap(QPixmap.fromImage(QImage(
            self.image_b.data,
            self.image_b.shape[1],
            self.image_b.shape[0],
            self.image_b.strides[0],
            QImage.Format_RGB888,
        )))
        self.centralWidget().widget().adjustSize()
        message = f'{len(contours)} particles detected.'
        if contours:
            avg = int(sum(map(cv2.contourArea, contours)) // len(contours))
            message += f' Average size: {avg}px.'
        self.statusBar().showMessage(message)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

