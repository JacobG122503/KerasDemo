import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
import gradio as gr
from model_definition import (
    TransformerEncoderBlock,
    PositionalEmbedding,
    TransformerDecoderBlock,
    custom_standardization,
    SEQ_LENGTH,
    VOCAB_SIZE,
    EMBED_DIM,
    FF_DIM,
)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGE_SIZE = (299, 299)
APP_CSS = """
footer {
    display: none !important;
}
"""

MODEL_CACHE = {}


def has_required_artifacts(model_dir):
    required = (
        "cnn_model.keras",
        "encoder_weights.npz",
        "decoder_weights.npz",
        "text_vectorization.keras",
    )
    return all(os.path.exists(os.path.join(model_dir, name)) for name in required)


def load_npz_weights(layer, weights_path):
    with np.load(weights_path) as weights_data:
        layer.set_weights([weights_data[key] for key in weights_data.files])


def get_available_model_dirs():
    if not os.path.exists(MODELS_DIR) or not os.listdir(MODELS_DIR):
        raise FileNotFoundError("No trained models found in 'models' directory. Please run train.py first.")

    model_dirs = [
        d for d in os.listdir(MODELS_DIR)
        if d.startswith("e")
        and os.path.isdir(os.path.join(MODELS_DIR, d))
        and has_required_artifacts(os.path.join(MODELS_DIR, d))
    ]
    if not model_dirs:
        raise FileNotFoundError("No usable 'e*' model directories found in 'models'. Please run training first.")

    return sorted(model_dirs, key=lambda d: int(d[1:]), reverse=True)


def load_model_bundle(model_dir_name):
    if model_dir_name in MODEL_CACHE:
        return MODEL_CACHE[model_dir_name]

    model_path = os.path.join(MODELS_DIR, model_dir_name)
    print(f"Loading models from: {model_path}")

    custom_objects = {
        "TransformerEncoderBlock": TransformerEncoderBlock,
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerDecoderBlock": TransformerDecoderBlock,
        "custom_standardization": custom_standardization,
    }
    with keras.saving.custom_object_scope(custom_objects):
        cnn_model = keras.models.load_model(os.path.join(model_path, "cnn_model.keras"))
        vectorization_model = keras.models.load_model(os.path.join(model_path, "text_vectorization.keras"))

    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)

    dummy_image = tf.zeros((1, *IMAGE_SIZE, 3), dtype=tf.float32)
    dummy_img_embed = cnn_model(dummy_image, training=False)
    dummy_encoder_out = encoder(dummy_img_embed, training=False)
    dummy_tokens = tf.zeros((1, SEQ_LENGTH - 1), dtype=tf.int32)
    decoder(dummy_tokens, dummy_encoder_out, training=False, mask=tf.not_equal(dummy_tokens, 0))

    load_npz_weights(encoder, os.path.join(model_path, "encoder_weights.npz"))
    load_npz_weights(decoder, os.path.join(model_path, "decoder_weights.npz"))

    vectorization = vectorization_model.layers[0]
    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    token_lookup = {token: index for index, token in index_lookup.items()}

    bundle = {
        "cnn_model": cnn_model,
        "encoder": encoder,
        "decoder": decoder,
        "index_lookup": index_lookup,
        "token_lookup": token_lookup,
    }
    MODEL_CACHE[model_dir_name] = bundle
    return bundle

model_dirs = get_available_model_dirs()
max_decoded_sentence_length = SEQ_LENGTH - 1

def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def generate_caption(model_dir_name, image):
    if image is None:
        return "Please upload an image."

    bundle = load_model_bundle(model_dir_name)
    cnn_model = bundle["cnn_model"]
    encoder = bundle["encoder"]
    decoder = bundle["decoder"]
    index_lookup = bundle["index_lookup"]
    token_lookup = bundle["token_lookup"]

    # Process the image
    img = tf.convert_to_tensor(np.array(image))
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, 0)
    
    # Pass the image through the CNN
    img_embed = cnn_model(img, training=False)
    
    # Pass the image features to the Transformer encoder
    encoder_out = encoder(img_embed, training=False)

    # Generate the caption using the Transformer decoder
    start_token = token_lookup["<start>"]
    end_token = token_lookup["<end>"]

    decoded_sentence = [start_token]
    
    for i in range(max_decoded_sentence_length):
        tokenized_sentence = np.zeros((1, SEQ_LENGTH - 1), dtype=np.int32)
        tokenized_sentence[0, : len(decoded_sentence)] = decoded_sentence
        token_mask = tokenized_sentence != 0
        
        # Get the next token prediction
        predictions = decoder(tokenized_sentence, encoder_out, training=False, mask=token_mask)
        
        # Sample the next token
        sampled_token_index = int(np.argmax(predictions[0, len(decoded_sentence) - 1, :]))
        
        # Convert the token index to a word
        sampled_token = index_lookup[sampled_token_index]
        
        if sampled_token == "<end>":
            break
            
        decoded_sentence.append(sampled_token_index)

    # Convert the token indices back to words
    caption_tokens = [index_lookup[i] for i in decoded_sentence[1:] if index_lookup[i] not in {"<start>", "<end>"}]
    caption = " ".join(caption_tokens)
    return caption


iface = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Dropdown(choices=model_dirs, value=model_dirs[0], label="Model Checkpoint"),
        gr.Image(type="pil", label="Image"),
    ],
    outputs=gr.Textbox(label="Caption"),
    title="Image Captioning with Keras",
    description="Upload an image and switch between trained checkpoints to compare captions.",
    css=APP_CSS,
)

if __name__ == "__main__":
    iface.launch()
