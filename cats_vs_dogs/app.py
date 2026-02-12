import sys
import os
import io
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox, QFrame)
from PyQt6.QtGui import QPixmap, QFont, QImage, QShortcut, QKeySequence
from PyQt6.QtCore import Qt, QBuffer, QIODevice

class CatsDogsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cat vs Dog Classifier")
        self.setFixedSize(450, 600)

        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Title
        self.label_title = QLabel("Cat or Dog?")
        self.label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        layout.addWidget(self.label_title)

        # Image Preview Container
        self.frame = QFrame()
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setStyleSheet("background-color: #f0f0f0; border-radius: 10px; border: 2px dashed #cccccc;")
        frame_layout = QVBoxLayout(self.frame)
        
        self.label_image = QLabel("No Image Selected")
        self.label_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_image.setStyleSheet("border: none; color: #888888;")
        self.label_image.setFixedSize(360, 360) 
        self.label_image.setScaledContents(True)
        
        # Centering the image in the frame
        frame_layout.addWidget(self.label_image, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.frame)

        # Result Label
        self.label_result = QLabel("Waiting for image...")
        self.label_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_result.setFont(QFont("Arial", 18))
        self.label_result.setStyleSheet("color: #555555;")
        layout.addWidget(self.label_result)

        # Buttons Layout
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        layout.addLayout(btn_layout)

        # Button Style
        btn_style = """
            QPushButton {
                background-color: #007AFF;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0062CC;
            }
            QPushButton:disabled {
                background-color: #cccccc; 
                color: #666666;
            }
        """

        # Select Image Button
        self.btn_load = QPushButton("Select Image")
        self.btn_load.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load.setStyleSheet(btn_style)
        self.btn_load.clicked.connect(self.load_image_from_file)
        btn_layout.addWidget(self.btn_load)

        # Paste Button
        self.btn_paste = QPushButton("Paste Image")
        self.btn_paste.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_paste.setStyleSheet(btn_style)
        self.btn_paste.clicked.connect(self.paste_from_clipboard)
        btn_layout.addWidget(self.btn_paste)

        # Keyboard shortcut: Ctrl+V / Cmd+V
        self.shortcut_paste = QShortcut(QKeySequence.StandardKey.Paste, self)
        self.shortcut_paste.activated.connect(self.paste_from_clipboard)

        # Load Model
        self.model = None
        # Defer model loading to show UI first if needed, but here simple is fine
        self.load_model()

    def load_model(self):
        # We need to change directory to where the script is or use absolute path
        # But assuming user runs from workspace root, we check cats_vs_dogs folder
        model_path = os.path.join(os.path.dirname(__file__), "cats_dogs_model.keras")
        
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                print("Model loaded successfully.")
            except Exception as e:
                self.label_result.setText("Error loading model.")
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
        else:
            self.label_result.setText("Model not found!")
            QMessageBox.warning(self, "Model Missing", 
                "Could not find 'cats_dogs_model.keras'.\n\n"
                "Please run the 'image_classification.ipynb' notebook first to train and save the model.")
            self.btn_load.setEnabled(False)
            self.btn_paste.setEnabled(False)

    def load_image_from_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            # Display
            pixmap = QPixmap(file_name)
            self.display_pixmap(pixmap)
            
            # Predict
            try:
                # Load as PIL
                img = keras.utils.load_img(file_name, target_size=(180, 180))
                self.predict_image(img)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def paste_from_clipboard(self):
        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()
        
        if mime.hasImage():
            qimg = clipboard.image()
            if not qimg.isNull():
                # Display
                pixmap = QPixmap.fromImage(qimg)
                self.display_pixmap(pixmap)
                
                # Convert QImage to PIL Image
                buffer = QBuffer()
                buffer.open(QIODevice.OpenModeFlag.ReadWrite)
                qimg.save(buffer, "PNG")
                pil_img = Image.open(io.BytesIO(bytes(buffer.data())))
                
                # Resize and convert to RGB (remove alpha if present)
                pil_img = pil_img.convert("RGB")
                pil_img = pil_img.resize((180, 180))
                
                self.predict_image(pil_img)
        else:
            QMessageBox.information(self, "Info", "No image found in clipboard!")

    def display_pixmap(self, pixmap):
        w = self.label_image.width()
        h = self.label_image.height()
        self.label_image.setPixmap(pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.label_image.setStyleSheet("border: none;")

    def predict_image(self, img_pil):
        # 2. Predict
        if self.model:
            self.label_result.setText("Thinking...")
            QApplication.processEvents() # Force UI update
            
            try:
                # Preprocess
                img_array = keras.utils.img_to_array(img_pil)
                img_array = np.expand_dims(img_array, 0) # Create batch axis

                # Get raw prediction (logit)
                predictions = self.model.predict(img_array)
                logit = predictions[0][0]
                
                # Apply Sigmoid manually to be safe across TF versions
                score = 1 / (1 + np.exp(-logit))
                
                # Tutorial Logic breakdown:
                # The model is trained with "Dog" folder and "Cat" folder. 
                # Usually alphabetical: 0=Cat, 1=Dog.
                # Score 0.0 -> Cat, Score 1.0 -> Dog
                
                cat_confidence = 100 * (1 - score)
                dog_confidence = 100 * score
                
                if dog_confidence > cat_confidence:
                    self.label_result.setText(f"It's a DOG! 🐶 ({dog_confidence:.1f}%)")
                    self.label_result.setStyleSheet("color: #D35400; font-weight: bold; font-size: 20px;")
                else:
                    self.label_result.setText(f"It's a CAT! 🐱 ({cat_confidence:.1f}%)")
                    self.label_result.setStyleSheet("color: #27AE60; font-weight: bold; font-size: 20px;")
                    
            except Exception as e:
                 self.label_result.setText("Error")
                 QMessageBox.critical(self, "Prediction Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CatsDogsApp()
    window.show()
    sys.exit(app.exec())
