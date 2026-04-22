import os
import argparse
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread
from skimage.transform import resize

os.environ["KERAS_BACKEND"] = "jax"
import keras
import keras_hub

label_map = {"Contradictory": 0, "Implies": 1, "NoEntailment": 2}
inv_label_map = {v: k for k, v in label_map.items()}

def main():
    parser = argparse.ArgumentParser(description="Run inference with the trained Multimodal model.")
    parser.add_argument("--model_path", type=str, help="Path to the trained .keras model file")
    args = parser.parse_args()

    model_path = args.model_path

    if not model_path:
        models_dir = "models"
        available_models = []
        if os.path.exists(models_dir):
            available_models = glob.glob(os.path.join(models_dir, "*.keras"))
        
        if not available_models:
            print("No trained models found in the 'models' directory.")
            print("Please finish training a model first, or specify the path with --model_path.")
            return

        print("Available models:")
        for i, model_file in enumerate(available_models):
            print(f"[{i+1}] {model_file}")
        
        while True:
            try:
                choice = input("\nEnter the number of the model you want to load (or 'q' to quit): ")
                if choice.lower() == 'q':
                    return
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_models):
                    model_path = available_models[choice_idx]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'")
        return

    print(f"\nLoading model from {model_path}...")
    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\nModel loaded successfully! Fetching sample data from the dataset...")
    image_base_path = keras.utils.get_file(
        "tweet_images",
        "https://github.com/sayakpaul/Multimodal-Entailment-Baseline/releases/download/v1.0.0/tweet_images.tar.gz",
        untar=True,
    )
    df = pd.read_csv("https://github.com/sayakpaul/Multimodal-Entailment-Baseline/raw/main/csvs/tweets.csv").iloc[0:1000]

    text_preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset("bert_base_en_uncased", sequence_length=128)
    bert_input_features = ["padding_mask", "segment_ids", "token_ids"]
    
    def process_img(file_name):
        img = resize(imread(file_name), (128, 128))
        if img.shape[2] == 4:
            return np.array(Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")) / 255.0
        return img

    while True:
        try:
            choice = input("\nPress Enter to evaluate a random example (or type 'q' to quit): ")
            if choice.lower() == 'q':
                break
            
            idx = random.choice(range(len(df)))
            row = df.iloc[idx]
            
            id_1 = row["id_1"]
            id_2 = row["id_2"]
            ext_1 = row["image_1"].split(".")[-1]
            ext_2 = row["image_2"].split(".")[-1]
            
            img1_path = os.path.join(image_base_path, str(id_1) + f".{ext_1}")
            img2_path = os.path.join(image_base_path, str(id_2) + f".{ext_2}")
            
            text_1 = row["text_1"]
            text_2 = row["text_2"]
            true_label = row["label"]
            
            print("\n--- Example Input ---")
            print(f"Text 1: {text_1}")
            print(f"Text 2: {text_2}")
            print(f"True Label: {true_label}")
            
            img1 = np.expand_dims(process_img(img1_path), axis=0)
            img2 = np.expand_dims(process_img(img2_path), axis=0)
            
            processed_text = text_preprocessor([text_1, text_2])
            text_inputs = {
                feature: np.expand_dims(keras.ops.reshape(processed_text[feature], [-1]), axis=0)
                for feature in bert_input_features
            }
            
            inputs = {
                "image_1": img1,
                "image_2": img2,
                **text_inputs
            }
            
            print("\nPredicting...")
            prediction = model.predict(inputs, verbose=0)
            predicted_class_idx = np.argmax(prediction[0])
            predicted_label = inv_label_map[predicted_class_idx]
            
            print(f"Predicted Label: {predicted_label}")
            print(f"Confidence: {prediction[0][predicted_class_idx] * 100:.2f}%")
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
