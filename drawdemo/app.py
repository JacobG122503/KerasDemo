import sys
import os
import numpy as np
import cv2
import typing

# TensorFlow / Keras
from tensorflow import keras

# PyQt6
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QProgressBar, QFrame)
from PyQt6.QtGui import QPainter, QPen, QImage, QColor, QFont, QPixmap
from PyQt6.QtCore import Qt, QPoint, QRect

# --- Model Loading ---
if not os.path.exists('mnist_model.keras'):
    print("Error: 'mnist_model.keras' not found!")
    sys.exit(1)

print("Loading model...")
model = keras.models.load_model('mnist_model.keras')
print("Model loaded.")

# --- Drawing Widget ---
class DrawingCanvas(QWidget):
    def __init__(self, parent=None, on_change_callback=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        
        self.drawing = False
        self.last_point = QPoint()
        self.pen_width = 16
        self.pen_color = Qt.GlobalColor.black
        
        self.on_change_callback = on_change_callback

    def clear(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update()
        if self.on_change_callback:
            self.on_change_callback()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()
            self.draw_point(event.position().toPoint())

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            self.draw_line(event.position().toPoint())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            if self.on_change_callback:
                self.on_change_callback()

    def draw_point(self, point):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPoint(point)
        self.update()
        self.last_point = point
        # Trigger update immediately while drawing
        if self.on_change_callback:
            self.on_change_callback()

    def draw_line(self, point):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(self.last_point, point)
        self.update()
        self.last_point = point
        # Trigger update immediately while drawing
        if self.on_change_callback:
            self.on_change_callback()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = event.rect()
        painter.drawImage(rect, self.image, rect)

    def get_image_array(self):
        # Convert QImage to numpy array
        # 1. Get raw data
        size = self.image.size()
        s = self.image.bits().asstring(size.width() * size.height() * 4)
        # 2. From buffer to array
        arr = np.frombuffer(s, dtype=np.uint8).reshape((size.height(), size.width(), 4))
        # 3. Drop alpha (Format_RGB32 has XXRRGGBB, but QImage often pads differently. 
        #    Actually RGB32 is 0xffRRGGBB.
        return arr[:, :, :3] # Keep RGB

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handwritten Digit Recognizer")
        
        # Central Widget & Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # --- Left Column: Canvas ---
        left_col = QVBoxLayout()
        
        # Canvas Container (border)
        canvas_container = QFrame()
        canvas_container.setFrameShape(QFrame.Shape.Box)
        canvas_container.setLineWidth(1)
        cc_layout = QVBoxLayout(canvas_container)
        cc_layout.setContentsMargins(1, 1, 1, 1) # Thin border
        
        self.canvas = DrawingCanvas(on_change_callback=self.predict_digit)
        cc_layout.addWidget(self.canvas)
        
        left_col.addWidget(canvas_container)
        
        # Instructions
        instr = QLabel("Draw a digit (0-9)")
        instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_col.addWidget(instr)
        
        # Clear Button
        btn = QPushButton("Clear Canvas")
        btn.clicked.connect(self.canvas.clear)
        left_col.addWidget(btn)
        
        layout.addLayout(left_col)
        
        # --- Right Column: Predictions ---
        right_col = QVBoxLayout()
        
        # Big Guess Label
        lbl_title = QLabel("I think it is a:")
        lbl_title.setFont(QFont("Arial", 16))
        right_col.addWidget(lbl_title)
        
        self.lbl_guess = QLabel("-")
        self.lbl_guess.setFont(QFont("Arial", 64, QFont.Weight.Bold))
        self.lbl_guess.setStyleSheet("color: blue")
        right_col.addWidget(self.lbl_guess)
        
        # Progress Bars (Top 3)
        self.bars = []
        for i in range(3):
            bar_layout = QVBoxLayout()
            lbl = QLabel(f"")
            lbl.setFont(QFont("Courier New", 12))
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setTextVisible(False)
            
            bar_layout.addWidget(lbl)
            bar_layout.addWidget(progress)
            
            right_col.addLayout(bar_layout)
            self.bars.append((lbl, progress))
            
        right_col.addStretch() # Push everything up
        layout.addLayout(right_col)

    def predict_digit(self):
        # 1. Get image from canvas
        image_rgb = self.canvas.get_image_array()
        
        # 2. Preprocessing
        # Convert to Grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Invert (White on Black for MNIST)
        # QImage white is 255. We want 0 for background.
        gray = 255 - gray
        
        # Smart Crop
        coords = cv2.findNonZero(gray)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            digit = gray[y:y+h, x:x+w]
            
            # Resize fit 20x20
            if w > h:
                new_w = 20
                new_h = int(h * (20 / w))
            else:
                new_h = 20
                new_w = int(w * (20 / h))
            
            resized_digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Paste into 28x28 Center
            final_image = np.zeros((28, 28), dtype=np.uint8)
            pad_top = (28 - new_h) // 2
            pad_left = (28 - new_w) // 2
            final_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_digit
            
            input_data = final_image.astype('float32') / 255.0
            input_data = input_data.reshape(1, 28, 28)
            
            # 3. Predict
            prediction = model.predict(input_data, verbose=0)[0]
            
            # 4. Update UI
            top_3_indices = prediction.argsort()[-3:][::-1]
            
            # Main Guess
            self.lbl_guess.setText(str(top_3_indices[0]))
            
            # Top 3 Bars
            for i in range(3):
                idx = top_3_indices[i]
                score = prediction[idx] * 100
                lbl, bar = self.bars[i]
                
                lbl.setText(f"Digit {idx}: {score:.1f}%")
                bar.setValue(int(score))
                
        else:
            # Empty
            self.lbl_guess.setText("-")
            for lbl, bar in self.bars:
                lbl.setText("")
                bar.setValue(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
