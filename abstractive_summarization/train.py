import os
import argparse
import time
import sys

# Argument parsing
parser = argparse.ArgumentParser(description="Abstractive Summarization with BART")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train (e.g. 10 or 20)")
args = parser.parse_args()

import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds
import py7zr

# Hyperparameters
BATCH_SIZE = 8
NUM_BATCHES = 600
EPOCHS = args.epochs
MAX_ENCODER_SEQUENCE_LENGTH = 512
MAX_DECODER_SEQUENCE_LENGTH = 128
MAX_GENERATION_LENGTH = 40

print(f"--- Starting training for {EPOCHS} epochs ---")

# --- DATASET PREPARATION ---
print("Downloading and extracting SAMSum dataset...")
filename = keras.utils.get_file(
    "corpus.7z",
    origin="https://arxiv.org/src/1911.12237v2/anc/corpus.7z",
)

extract_path = os.path.expanduser("~/tensorflow_datasets/downloads/manual")
os.makedirs(extract_path, exist_ok=True)
with py7zr.SevenZipFile(filename, mode="r") as z:
    z.extractall(path=extract_path)

print("Loading dataset...")
samsum_ds = tfds.load("samsum", split="train", as_supervised=True)

train_ds = (
    samsum_ds.map(
        lambda dialogue, summary: {"encoder_text": dialogue, "decoder_text": summary}
    )
    .batch(BATCH_SIZE)
    .cache()
)
train_ds = train_ds.take(NUM_BATCHES)

# --- MODEL SETUP ---
print("Setting up BART model...")
preprocessor = keras_hub.models.BartSeq2SeqLMPreprocessor.from_preset(
    "bart_base_en",
    encoder_sequence_length=MAX_ENCODER_SEQUENCE_LENGTH,
    decoder_sequence_length=MAX_DECODER_SEQUENCE_LENGTH,
)
bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset(
    "bart_base_en", preprocessor=preprocessor
)

optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
    epsilon=1e-6,
    global_clipnorm=1.0,
)

optimizer.exclude_from_weight_decay(var_names=["bias"])
optimizer.exclude_from_weight_decay(var_names=["gamma"])
optimizer.exclude_from_weight_decay(var_names=["beta"])

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bart_lm.compile(
    optimizer=optimizer,
    loss=loss,
    weighted_metrics=["accuracy"],
)

# --- TRAINING LOOP ---
print(f"Training model for {EPOCHS} epochs...")
# Keras automatically displays a progress bar with an ETA in interactive mode
bart_lm.fit(train_ds, epochs=EPOCHS)

os.makedirs("models", exist_ok=True)
model_path = f"models/bart_summarization_{EPOCHS}_epochs.keras"
bart_lm.save(model_path)
print(f"Model saved to {model_path}")

# --- GENERATION & EVALUATION ---
print("Testing generation...")
def generate_text(model, input_text, max_length=200, print_time_taken=False):
    start = time.time()
    output = model.generate(input_text, max_length=max_length)
    end = time.time()
    if print_time_taken:
        print(f"Total Time Elapsed: {end - start:.2f}s")
    return output

val_ds = tfds.load("samsum", split="validation", as_supervised=True)
val_ds = val_ds.take(1)

# Warm up XLA
_ = generate_text(bart_lm, "sample text", max_length=MAX_GENERATION_LENGTH)

generated_summaries = generate_text(
    bart_lm,
    val_ds.map(lambda dialogue, _: dialogue).batch(1),
    max_length=MAX_GENERATION_LENGTH,
    print_time_taken=True,
)
print("Validation example generated:")
print(generated_summaries)
