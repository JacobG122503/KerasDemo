import keras
from keras import layers
import numpy as np
import os

# Parameters for the model and dataset.
TRAINING_SIZE = 50000
REVERSE = True
BATCH_SIZE = 32
TRAIN_DIGITS = 3
EPOCH_MILESTONES = [10, 30, 60]

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

print(f"Generating data for {TRAIN_DIGITS}-digit addition...")
local_maxlen = TRAIN_DIGITS + 1 + TRAIN_DIGITS
questions = []
expected = []
seen = set()

while len(questions) < TRAINING_SIZE:
    f = lambda: int("".join(np.random.choice(list("0123456789")) for i in range(np.random.randint(1, TRAIN_DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen: continue
    seen.add(key)
    q = "{}+{}".format(a, b)
    query = q + " " * (local_maxlen - len(q))
    ans = str(a + b)
    ans += " " * (TRAIN_DIGITS + 1 - len(ans))
    if REVERSE: query = query[::-1]
    questions.append(query)
    expected.append(ans)

print("Vectorizing data...")
x = np.zeros((len(questions), local_maxlen, len(chars)), dtype=bool)
y = np.zeros((len(questions), TRAIN_DIGITS + 1, len(chars)), dtype=bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, local_maxlen)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, TRAIN_DIGITS + 1)

print("Building model...")
model = keras.Sequential()
model.add(layers.Input((local_maxlen, len(chars))))
model.add(layers.LSTM(128))
model.add(layers.RepeatVector(TRAIN_DIGITS + 1))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.Dense(len(chars), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

previous_epoch = 0
for target_epoch in EPOCH_MILESTONES:
    print(f"\n--- Training model up to {target_epoch} epochs ---")
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=target_epoch, initial_epoch=previous_epoch, validation_split=0.1, verbose=1)
    
    filename = f"model_{TRAIN_DIGITS}digits_{target_epoch}epochs.keras"
    path = os.path.join(MODELS_DIR, filename)
    model.save(path)
    print(f"Saved checkpoint: {filename}")
    
    previous_epoch = target_epoch

print("All training complete!")