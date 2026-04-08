import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_hub
import keras
import tensorflow as tf
import numpy as np
import gradio as gr

keras.utils.set_random_seed(42)

# --- Config ---
MAX_SEQUENCE_LENGTH = 512
VOCAB_SIZE = 15000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.txt")
FNET_PATH = os.path.join(MODELS_DIR, "fnet_classifier.keras")
TRANSFORMER_PATH = os.path.join(MODELS_DIR, "transformer_classifier.keras")

APP_CSS = """
footer { display: none !important; }
"""

# --- Load tokenizer and models ---
tokenizer = None
fnet_model = None
transformer_model = None


def load_artifacts():
    global tokenizer, fnet_model, transformer_model

    if not os.path.exists(VOCAB_PATH):
        return False, "vocab.txt not found. Please run train.py first."
    if not os.path.exists(FNET_PATH):
        return False, "fnet_classifier.keras not found. Please run train.py first."
    if not os.path.exists(TRANSFORMER_PATH):
        return False, "transformer_classifier.keras not found. Please run train.py first."

    print("Loading vocabulary...")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f]

    tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
        vocabulary=vocab,
        lowercase=False,
        sequence_length=MAX_SEQUENCE_LENGTH,
    )

    print("Loading FNet model...")
    fnet_model = keras.saving.load_model(FNET_PATH)

    print("Loading Transformer model...")
    transformer_model = keras.saving.load_model(TRANSFORMER_PATH)

    print("All models loaded.")
    return True, ""


def classify(review_text: str):
    if not review_text.strip():
        return "Please enter a review.", "", ""

    tokens = tokenizer(tf.constant([review_text.lower()]))
    input_tensor = {"input_ids": tokens}

    fnet_score = float(fnet_model.predict(input_tensor, verbose=0)[0][0])
    transformer_score = float(transformer_model.predict(input_tensor, verbose=0)[0][0])

    def fmt(score):
        label = "POSITIVE" if score >= 0.5 else "NEGATIVE"
        confidence = score if score >= 0.5 else 1.0 - score
        return f"{label} ({confidence * 100:.1f}% confident)"

    return fmt(fnet_score), fmt(transformer_score), _bar(fnet_score, transformer_score)


def _bar(fnet_score, transformer_score):
    def bar(score):
        filled = int(round(score * 20))
        empty = 20 - filled
        return f"{'█' * filled}{'░' * empty}  {score * 100:.1f}%"

    return (
        f"FNet       : {bar(fnet_score)}\n"
        f"Transformer: {bar(transformer_score)}"
    )


# --- Load on startup ---
ok, err_msg = load_artifacts()

with gr.Blocks(css=APP_CSS, title="FNet Text Classification") as demo:
    gr.Markdown("# FNet vs Transformer — IMDb Sentiment Classifier")
    gr.Markdown(
        "Type or paste a movie review below and compare predictions from an **FNet** "
        "classifier and a standard **Transformer** classifier, both trained on the IMDb dataset."
    )

    if not ok:
        gr.Markdown(f"⚠️ **{err_msg}**")
    else:
        with gr.Row():
            review_input = gr.Textbox(
                label="Movie Review",
                placeholder="e.g. This movie was absolutely fantastic! Great story and acting.",
                lines=6,
            )

        classify_btn = gr.Button("Classify", variant="primary")

        with gr.Row():
            fnet_output = gr.Textbox(label="FNet Prediction", interactive=False)
            transformer_output = gr.Textbox(label="Transformer Prediction", interactive=False)

        score_bars = gr.Textbox(
            label="Confidence Scores (positive sentiment probability)",
            interactive=False,
            lines=3,
        )

        classify_btn.click(
            fn=classify,
            inputs=[review_input],
            outputs=[fnet_output, transformer_output, score_bars],
        )

        gr.Examples(
            examples=[
                ["This film is a masterpiece. The performances are outstanding and the story kept me on the edge of my seat the entire time."],
                ["Absolutely terrible. The plot made no sense, the acting was wooden, and I wanted to leave after 10 minutes."],
                ["A decent watch for a lazy afternoon but nothing memorable. Some good moments but largely forgettable."],
            ],
            inputs=[review_input],
        )

        gr.Markdown(
            "**About**: FNet replaces the self-attention mechanism with a Fourier Transform, "
            "achieving ~92–97% of BERT's accuracy while being ~80% faster on GPUs."
        )

if __name__ == "__main__":
    demo.launch()
