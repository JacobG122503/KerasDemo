import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import re
import pickle
import numpy as np
import tensorflow as tf
import keras
from keras.layers import TextVectorization
import gradio as gr
from model_definition import (
    TransformerEncoderBlock,
    PositionalEmbedding,
    TransformerDecoderBlock,
    custom_standardization,
    SEQ_LENGTH,
)

# Constants
IMAGE_SIZE = (299, 299)

# Find the model directory with the highest epoch number
if not os.path.exists("models") or not os.listdir("models"):
    raise FileNotFoundError("No trained models found in 'models' directory. Please run train.py first.")

model_dirs = [d for d in os.listdir("models") if d.startswith('e') and os.path.isdir(os.path.join("models", d))]
if not model_dirs:
    raise FileNotFoundError("No 'e*' model directories found in 'models'. Please run train.py first.")

# Sort by epoch number (e.g., 'e100' > 'e10') and get the latest
latest_model_dir = sorted(model_dirs, key=lambda d: int(d[1:]), reverse=True)[0]
model_path = os.path.join("models", latest_model_dir)
print(f"Loading models from: {model_path}")

# Load the models and vectorization layer
custom_objects = {
    "TransformerEncoderBlock": TransformerEncoderBlock,
    "PositionalEmbedding": PositionalEmbedding,
    "TransformerDecoderBlock": TransformerDecoderBlock,
    "custom_standardization": custom_standardization,
}
with keras.saving.custom_object_scope(custom_objects):
    cnn_model = keras.models.load_model(os.path.join(model_path, "cnn_model"))
    encoder = keras.models.load_model(os.path.join(model_path, "encoder"))
    decoder = keras.models.load_model(os.path.join(model_path, "decoder"))
    vectorization_model = keras.models.load_model(os.path.join(model_path, "text_vectorization"))

vectorization = vectorization_model.layers[0]
vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1

def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def generate_caption(image):
    # Process the image
    img = tf.image.resize(image, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, 0)
    
    # Pass the image through the CNN
    img_embed = cnn_model(img, training=False)
    
    # Pass the image features to the Transformer encoder
    encoder_out = encoder(img_embed, training=False)

    # Generate the caption using the Transformer decoder
    start_token = vectorization(["<start>"])[0].numpy().item()
    end_token = vectorization(["<end>"])[0].numpy().item()

    decoded_sentence = [start_token]
    
    for i in range(max_decoded_sentence_length):
        tokenized_sentence = np.array([decoded_sentence])
        
        # Get the next token prediction
        predictions = decoder([tokenized_sentence, encoder_out], training=False)
        
        # Sample the next token
        sampled_token_index = np.argmax(predictions[0, i, :])
        
        # Convert the token index to a word
        sampled_token = index_lookup[sampled_token_index]
        
        if sampled_token == "<end>":
            break
            
        decoded_sentence.append(sampled_token_index)

    # Convert the token indices back to words
    caption = " ".join([index_lookup[i] for i in decoded_sentence[1:]])
    return caption


# Create the Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning with Keras",
    description="Upload an image and see the generated caption.",
)

if __name__ == "__main__":
    iface.launch()
