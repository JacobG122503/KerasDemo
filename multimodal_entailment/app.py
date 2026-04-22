import os
import argparse
import glob
import random
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from skimage.io import imread
from skimage.transform import resize

os.environ["KERAS_BACKEND"] = "jax"
import keras
import keras_hub

label_map = {"Contradictory": 0, "Implies": 1, "NoEntailment": 2}
inv_label_map = {v: k for k, v in label_map.items()}

class EntailmentApp:
    def __init__(self, root, model_path=None):
        self.root = root
        self.root.title("Multimodal Entailment")
        self.root.geometry("1100x800")
        
        self.model = None
        self.df = None
        self.image_base_path = None
        self.text_preprocessor = None
        self.bert_input_features = ["padding_mask", "segment_ids", "token_ids"]
        
        self.setup_ui()
        
        # Determine available models
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(script_dir, "models")
        self.available_models = []
        if os.path.exists(self.models_dir):
            self.available_models = glob.glob(os.path.join(self.models_dir, "*.keras"))
            
        if model_path and os.path.exists(model_path):
            self.root.after(100, lambda: self.load_model(model_path))
        else:
            self.root.after(100, self.prompt_model_selection)

    def setup_ui(self):
        # Top Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing GUI...")
        status_label = tk.Label(control_frame, textvariable=self.status_var, font=("Helvetica", 14, "bold"))
        status_label.pack(side=tk.LEFT)
        
        self.next_btn = tk.Button(control_frame, text="Generate Random Example ➔", command=self.load_random_example, state=tk.DISABLED, font=("Helvetica", 14), bg="#007bff", fg="black")
        self.next_btn.pack(side=tk.RIGHT)
        
        # Main content frame
        content_frame = tk.Frame(self.root)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel for Tweet 1
        left_frame = tk.LabelFrame(content_frame, text="Tweet 1", font=("Helvetica", 14, "bold"), padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.img1_label = tk.Label(left_frame)
        self.img1_label.pack(pady=10)
        
        self.text1_var = tk.StringVar()
        self.text1_var.set("Loading...")
        text1_display = tk.Message(left_frame, textvariable=self.text1_var, width=450, font=("Helvetica", 13), justify=tk.CENTER)
        text1_display.pack(pady=10, fill=tk.X, expand=True)
        
        # Right panel for Tweet 2
        right_frame = tk.LabelFrame(content_frame, text="Tweet 2", font=("Helvetica", 14, "bold"), padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        self.img2_label = tk.Label(right_frame)
        self.img2_label.pack(pady=10)
        
        self.text2_var = tk.StringVar()
        self.text2_var.set("Loading...")
        text2_display = tk.Message(right_frame, textvariable=self.text2_var, width=450, font=("Helvetica", 13), justify=tk.CENTER)
        text2_display.pack(pady=10, fill=tk.X, expand=True)
        
        # Bottom frame for results
        result_frame = tk.Frame(self.root, pady=10)
        result_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)
        
        self.true_label_var = tk.StringVar()
        self.true_label_var.set("True Label: --")
        tk.Label(result_frame, textvariable=self.true_label_var, font=("Helvetica", 16), fg="blue").pack(pady=5)
        
        self.pred_label_var = tk.StringVar()
        self.pred_label_var.set("Predicted: --")
        self.pred_label_widget = tk.Label(result_frame, textvariable=self.pred_label_var, font=("Helvetica", 20, "bold"))
        self.pred_label_widget.pack(pady=5)

    def prompt_model_selection(self):
        if not self.available_models:
            messagebox.showerror("No Models", "No trained models found in the 'models' directory.\nPlease finish training a model first!")
            self.root.destroy()
            return
            
        top = tk.Toplevel(self.root)
        top.title("Select Model")
        top.geometry("450x150")
        top.transient(self.root)
        top.grab_set()
        
        tk.Label(top, text="Select a trained Keras model to load:", font=("Helvetica", 12)).pack(pady=15)
        
        combo = ttk.Combobox(top, values=[os.path.basename(m) for m in self.available_models], state="readonly", width=45, font=("Helvetica", 12))
        combo.current(0)
        combo.pack(pady=5)
        
        def on_select():
            selected_model = self.available_models[combo.current()]
            top.destroy()
            self.root.after(50, lambda: self.load_model(selected_model))
            
        tk.Button(top, text="Load Model", command=on_select, font=("Helvetica", 12, "bold")).pack(pady=15)

    def load_model(self, path):
        self.status_var.set(f"Loading Model: {os.path.basename(path)}...")
        self.root.update()
        
        try:
            self.model = keras.models.load_model(path)
            
            self.status_var.set("Downloading/Loading Dataset (Tweets & Images)...")
            self.root.update()
            
            image_base_path = keras.utils.get_file(
                "tweet_images",
                "https://github.com/sayakpaul/Multimodal-Entailment-Baseline/releases/download/v1.0.0/tweet_images.tar.gz",
                untar=True,
            )
            self.image_base_path = os.path.join(image_base_path, "tweet_images")
            
            self.df = pd.read_csv("https://github.com/sayakpaul/Multimodal-Entailment-Baseline/raw/main/csvs/tweets.csv").iloc[0:1000]
            
            self.status_var.set("Loading BERT Preprocessor...")
            self.root.update()
            self.text_preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset("bert_base_en_uncased", sequence_length=128)
            
            self.status_var.set("Ready! Click 'Generate Random Example' to start.")
            self.next_btn.config(state=tk.NORMAL)
            
            self.load_random_example()
            
        except Exception as e:
            messagebox.showerror("Error Loading", f"Failed to initialize the app:\n{e}")
            self.root.destroy()

    def process_img(self, file_name):
        img = resize(imread(file_name), (128, 128))
        if img.shape[2] == 4:
            return np.array(Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")) / 255.0
        return img

    def display_image(self, file_path, label_widget):
        try:
            img = Image.open(file_path)
            img.thumbnail((350, 350), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo)
            label_widget.image = photo
        except Exception as e:
            label_widget.config(text=f"Error loading image:\n{e}", image="")

    def load_random_example(self):
        if self.df is None or self.model is None:
            return
            
        self.status_var.set("Processing Image & Text Tensors...")
        self.next_btn.config(state=tk.DISABLED)
        self.root.update()
        
        idx = random.choice(range(len(self.df)))
        row = self.df.iloc[idx]
        
        id_1 = row["id_1"]
        id_2 = row["id_2"]
        ext_1 = row["image_1"].split(".")[-1]
        ext_2 = row["image_2"].split(".")[-1]
        
        img1_path = os.path.join(self.image_base_path, str(id_1) + f".{ext_1}")
        img2_path = os.path.join(self.image_base_path, str(id_2) + f".{ext_2}")
        
        text_1 = row["text_1"]
        text_2 = row["text_2"]
        true_label = row["label"]
        
        self.text1_var.set(text_1)
        self.text2_var.set(text_2)
        self.true_label_var.set(f"True Label: {true_label}")
        self.pred_label_var.set("Predicting...")
        self.pred_label_widget.config(fg="black")
        
        self.display_image(img1_path, self.img1_label)
        self.display_image(img2_path, self.img2_label)
        self.root.update()
        
        # Run through model
        img1 = np.expand_dims(self.process_img(img1_path), axis=0)
        img2 = np.expand_dims(self.process_img(img2_path), axis=0)
        
        processed_text = self.text_preprocessor([text_1, text_2])
        text_inputs = {
            feature: np.expand_dims(keras.ops.reshape(processed_text[feature], [-1]), axis=0)
            for feature in self.bert_input_features
        }
        
        inputs = {
            "image_1": img1,
            "image_2": img2,
            **text_inputs
        }
        
        prediction = self.model.predict(inputs, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_label = inv_label_map[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx] * 100
        
        pred_text = f"Model Prediction: {predicted_label} ({confidence:.1f}% Confidence)"
        self.pred_label_var.set(pred_text)
        
        if predicted_label == true_label:
            self.pred_label_widget.config(fg="green")
        else:
            self.pred_label_widget.config(fg="red")
            
        self.status_var.set("Ready")
        self.next_btn.config(state=tk.NORMAL)

def main():
    parser = argparse.ArgumentParser(description="Run GUI inference with the trained Multimodal model.")
    parser.add_argument("--model_path", type=str, help="Path to the trained .keras model file")
    args = parser.parse_args()

    root = tk.Tk()
    app = EntailmentApp(root, args.model_path)
    root.mainloop()

if __name__ == "__main__":
    main()
