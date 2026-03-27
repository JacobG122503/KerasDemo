import keras
from keras import layers
import numpy as np
import os
from tqdm import tqdm
import time
import tensorflow as tf

# --- GPU Configuration ---
# The following code block will check for available GPUs and, if found,
# configure TensorFlow to use them. This is crucial for performance on an HPC.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Found {len(gpus)} physical GPUs, configured {len(logical_gpus)} logical GPUs.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# -------------------------


# Parameters for the model and dataset.
TRAINING_SIZE = 50000
REVERSE = True
BATCH_SIZE = 128

TRAIN_DIGITS = 6
TOTAL_EPOCHS = 100


# Using mixed precision can speed up training on compatible GPUs.
keras.mixed_precision.set_global_policy("mixed_float16")


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class CharacterTable:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            if c in self.char_indices:
                x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[x] for x in x)



chars = "0123456789+ "
ctable = CharacterTable(chars)
local_maxlen = TRAIN_DIGITS + 1 + TRAIN_DIGITS


def data_generator(batch_size):
    """Generates batches of data for training."""
    while True:
        questions = []
        expected = []
        seen = set()

        while len(questions) < batch_size:
            f = lambda: int(
                "".join(
                    np.random.choice(list("0123456789"))
                    for i in range(np.random.randint(1, TRAIN_DIGITS + 1))
                )
            )
            a, b = f(), f()
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            q = f"{a}+{b}"
            query = q + " " * (local_maxlen - len(q))
            ans = str(a + b)
            ans += " " * (TRAIN_DIGITS + 1 - len(ans))
            if REVERSE:
                query = query[::-1]
            questions.append(query)
            expected.append(ans)
        
        x = np.zeros((batch_size, local_maxlen, len(chars)), dtype=bool)
        y = np.zeros((batch_size, TRAIN_DIGITS + 1, len(chars)), dtype=bool)
        for i, sentence in enumerate(questions):
            x[i] = ctable.encode(sentence, local_maxlen)
        for i, sentence in enumerate(expected):
            y[i] = ctable.encode(sentence, TRAIN_DIGITS + 1)
        
        yield x, y



dataset = tf.data.Dataset.from_generator(

    lambda: data_generator(BATCH_SIZE),

    output_signature=(

        tf.TensorSpec(shape=(BATCH_SIZE, local_maxlen, len(chars)), dtype=tf.bool),

        tf.TensorSpec(shape=(BATCH_SIZE, TRAIN_DIGITS + 1, len(chars)), dtype=tf.bool),

    ),

)

dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



print("Building model...")

model = keras.Sequential()

model.add(layers.Input((local_maxlen, len(chars))))

model.add(layers.LSTM(128))

model.add(layers.RepeatVector(TRAIN_DIGITS + 1))

model.add(layers.LSTM(128, return_sequences=True))

model.add(layers.Dense(len(chars), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()





class TqdmProgressCallback(keras.callbacks.Callback):

    """A Keras callback to render a progress bar for epochs using tqdm."""



    def on_train_begin(self, logs=None):

        self.epochs = self.params["epochs"]

        self.pbar = tqdm(total=self.epochs, desc="Epochs", unit="epoch")

        self.epoch_times = []



    def on_epoch_begin(self, epoch, logs=None):

        self.epoch_start_time = time.time()



    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}



        epoch_time = time.time() - self.epoch_start_time

        self.epoch_times.append(epoch_time)

        avg_time = sum(self.epoch_times) / len(self.epoch_times)

        remaining_epochs = self.params["epochs"] - (epoch + 1)

        eta_seconds = int(avg_time * remaining_epochs)



        mins, secs = divmod(eta_seconds, 60)

        hrs, mins = divmod(mins, 60)



        eta_str = ""

        if hrs > 0:

            eta_str = f"{hrs}h {mins}m {secs}s"

        elif mins > 0:

            eta_str = f"{mins}m {secs}s"

        else:

            eta_str = f"{secs}s"



        self.pbar.update(1)

        self.pbar.set_postfix(

            {

                "loss": f"{logs.get('loss', 0):.4f}",

                "acc": f"{logs.get('accuracy', 0):.4f}",

                "val_loss": f"{logs.get('val_loss', 0):.4f}",

                "val_acc": f"{logs.get('val_accuracy', 0):.4f}",

                "ETA": eta_str,

            }

        )



    def on_train_end(self, logs=None):

        self.pbar.close()





print(f"\n--- Training model for {TOTAL_EPOCHS} epochs ---")

model.fit(

    dataset,

    steps_per_epoch=TRAINING_SIZE // BATCH_SIZE,

    epochs=TOTAL_EPOCHS,

    callbacks=[TqdmProgressCallback()],

    validation_data=dataset,

    validation_steps=TRAINING_SIZE // BATCH_SIZE // 10,

)



filename = f"model_{TRAIN_DIGITS}digits_{TOTAL_EPOCHS}epochs.keras"
path = os.path.join(MODELS_DIR, filename)
model.save(path)
print(f"\nSaved final model: {filename}")

print("\nTraining complete!")