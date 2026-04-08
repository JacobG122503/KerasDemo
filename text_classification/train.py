import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tarfile
import urllib.request
import shutil
import time

import keras_hub
import keras
import tensorflow as tf

keras.utils.set_random_seed(42)

# --- Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 3
MAX_SEQUENCE_LENGTH = 512
VOCAB_SIZE = 15000
EMBED_DIM = 128
INTERMEDIATE_DIM = 512
NUM_HEADS = 2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "aclImdb")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.txt")

os.makedirs(MODELS_DIR, exist_ok=True)


# --- Download and extract dataset if needed ---
def download_dataset():
    archive = os.path.join(BASE_DIR, "aclImdb_v1.tar.gz")
    if not os.path.exists(DATA_DIR):
        if not os.path.exists(archive):
            print("Downloading IMDB dataset (~80 MB)...")
            url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            urllib.request.urlretrieve(url, archive)
            print("Download complete.")
        print("Extracting dataset...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(BASE_DIR)
        print("Extraction complete.")

    unsup_dir = os.path.join(DATA_DIR, "train", "unsup")
    if os.path.exists(unsup_dir):
        shutil.rmtree(unsup_dir)
        print("Removed unlabelled unsup samples.")


download_dataset()


# --- Load datasets ---
print("\nLoading datasets...")
train_ds = keras.utils.text_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42,
)
val_ds = keras.utils.text_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42,
)
test_ds = keras.utils.text_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    batch_size=BATCH_SIZE,
)

train_ds = train_ds.map(lambda x, y: (tf.strings.lower(x), y))
val_ds = val_ds.map(lambda x, y: (tf.strings.lower(x), y))
test_ds = test_ds.map(lambda x, y: (tf.strings.lower(x), y))


# --- Train or load WordPiece vocabulary ---
reserved_tokens = ["[PAD]", "[UNK]"]

if os.path.exists(VOCAB_PATH):
    print(f"\nLoading existing vocabulary from {VOCAB_PATH} ...")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f]
else:
    print("\nTraining WordPiece vocabulary (this may take a few minutes)...")

    def train_word_piece(ds, vocab_size, reserved_tokens):
        word_piece_ds = ds.unbatch().map(lambda x, y: x)
        vocab = keras_hub.tokenizers.compute_word_piece_vocabulary(
            word_piece_ds.batch(1000).prefetch(2),
            vocabulary_size=vocab_size,
            reserved_tokens=reserved_tokens,
        )
        return vocab

    vocab = train_word_piece(train_ds, VOCAB_SIZE, reserved_tokens)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")
    print(f"Vocabulary saved to {VOCAB_PATH}")

tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    lowercase=False,
    sequence_length=MAX_SEQUENCE_LENGTH,
)


# --- Format datasets ---
def format_dataset(sentence, label):
    sentence = tokenizer(sentence)
    return ({"input_ids": sentence}, label)


def make_dataset(dataset):
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(512).prefetch(16).cache()


print("Tokenizing and caching datasets...")
train_ds = make_dataset(train_ds)
val_ds = make_dataset(val_ds)
test_ds = make_dataset(test_ds)


# --- Model builders ---
def build_fnet_classifier():
    input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")

    x = keras_hub.layers.TokenAndPositionEmbedding(
        vocabulary_size=VOCAB_SIZE,
        sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBED_DIM,
        mask_zero=True,
    )(input_ids)

    x = keras_hub.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
    x = keras_hub.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
    x = keras_hub.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.1)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(input_ids, outputs, name="fnet_classifier")


def build_transformer_classifier():
    input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")

    x = keras_hub.layers.TokenAndPositionEmbedding(
        vocabulary_size=VOCAB_SIZE,
        sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBED_DIM,
        mask_zero=True,
    )(input_ids)

    x = keras_hub.layers.TransformerEncoder(
        intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
    )(inputs=x)
    x = keras_hub.layers.TransformerEncoder(
        intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
    )(inputs=x)
    x = keras_hub.layers.TransformerEncoder(
        intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
    )(inputs=x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.1)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(input_ids, outputs, name="transformer_classifier")


# --- Train FNet ---
print("\n" + "=" * 50)
print("Training FNet Classifier")
print("=" * 50)

fnet_classifier = build_fnet_classifier()
fnet_classifier.summary()
fnet_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

fnet_start = time.time()
fnet_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
fnet_time = time.time() - fnet_start

print(f"\nFNet training time: {fnet_time:.1f}s")
print("FNet Test Evaluation:")
fnet_test = fnet_classifier.evaluate(test_ds)

fnet_path = os.path.join(MODELS_DIR, "fnet_classifier.keras")
fnet_classifier.save(fnet_path)
print(f"FNet model saved to {fnet_path}")


# --- Train Transformer ---
print("\n" + "=" * 50)
print("Training Transformer Classifier")
print("=" * 50)

transformer_classifier = build_transformer_classifier()
transformer_classifier.summary()
transformer_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

transformer_start = time.time()
transformer_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
transformer_time = time.time() - transformer_start

print(f"\nTransformer training time: {transformer_time:.1f}s")
print("Transformer Test Evaluation:")
transformer_test = transformer_classifier.evaluate(test_ds)

transformer_path = os.path.join(MODELS_DIR, "transformer_classifier.keras")
transformer_classifier.save(transformer_path)
print(f"Transformer model saved to {transformer_path}")


# --- Summary ---
print("\n" + "=" * 50)
print("Results Summary")
print("=" * 50)
print(f"{'Model':<25} {'Train Time':>12} {'Test Accuracy':>15}")
print("-" * 55)
print(f"{'FNet':<25} {fnet_time:>10.1f}s  {fnet_test[1] * 100:>13.2f}%")
print(f"{'Transformer':<25} {transformer_time:>10.1f}s  {transformer_test[1] * 100:>13.2f}%")
print(f"\nSpeedup (Transformer/FNet): {transformer_time / fnet_time:.2f}x")
print("\nAll models saved. Run app.py to try the classifiers.")
