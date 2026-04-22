import os
import argparse

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import keras_hub
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

BATCH_SIZE = 64
MIN_STRING_LEN = 512
SEQ_LEN = 128

EMBED_DIM = 256
FEED_FORWARD_DIM = 128
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000

print("Downloading dataset...")
extracted_path = keras.utils.get_file(
    origin="https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip",
    extract=True,
)
path = os.path.join(extracted_path, "simplebooks", "simplebooks-92-raw")

print("Loading dataset...")
raw_train_ds = (
    tf_data.TextLineDataset(os.path.join(path, "train.txt"))
    .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

raw_val_ds = (
    tf_data.TextLineDataset(os.path.join(path, "valid.txt"))
    .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
    .batch(BATCH_SIZE)
)

print("Computing vocabulary...")
vocab = keras_hub.tokenizers.compute_word_piece_vocabulary(
    raw_train_ds,
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,
    reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
)

tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
    lowercase=True,
)

start_packer = keras_hub.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)

def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels

train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(tf_data.AUTOTUNE)
val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(tf_data.AUTOTUNE)

print("Building model...")
inputs = keras.layers.Input(shape=(None,), dtype="int32")
embedding_layer = keras_hub.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=SEQ_LEN,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)
x = embedding_layer(inputs)

for _ in range(NUM_LAYERS):
    decoder_layer = keras_hub.layers.TransformerDecoder(
        num_heads=NUM_HEADS,
        intermediate_dim=FEED_FORWARD_DIM,
    )
    x = decoder_layer(x)

outputs = keras.layers.Dense(VOCAB_SIZE)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_hub.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])

print(f"Training for {args.epochs} epochs...")
model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

os.makedirs("models", exist_ok=True)
model_path = f"models/text_generation_{args.epochs}_epochs.keras"
model.save(model_path)
print(f"Model saved to {model_path}")
