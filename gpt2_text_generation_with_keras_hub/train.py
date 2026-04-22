import os
import argparse

os.environ["KERAS_BACKEND"] = "jax"

import keras_hub
import keras
import tensorflow as tf
import tensorflow_datasets as tfds

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

# Use global policy for performance
keras.mixed_precision.set_global_policy("mixed_float16")

print("Loading GPT-2 model...")
preprocessor = keras_hub.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_hub.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)

print("Loading Reddit TIFU dataset...")
reddit_ds = tfds.load("reddit_tifu", split="train", as_supervised=True)

train_ds = (
    reddit_ds.map(lambda document, _: document)
    .batch(32)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

train_ds = train_ds.take(500)

learning_rate = keras.optimizers.schedules.PolynomialDecay(
    5e-5,
    decay_steps=train_ds.cardinality() * args.epochs,
    end_learning_rate=0.0,
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
    weighted_metrics=["accuracy"],
)

print(f"Training for {args.epochs} epochs...")
gpt2_lm.fit(train_ds, epochs=args.epochs)

os.makedirs("models", exist_ok=True)
model_path = f"models/gpt2_keras_hub_{args.epochs}_epochs.keras"
gpt2_lm.save(model_path)
print(f"Model saved to {model_path}")
