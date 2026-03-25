import keras
from keras import layers
import numpy as np
import os
import time

# Parameters for the model and dataset.
TRAINING_SIZE = 50000
REVERSE = True

DIGITS = 6
EPOCHS = 60
BATCH_SIZE = 32

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
MAXLEN = DIGITS + 1 + DIGITS
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILENAME = f"model_{DIGITS}digits_{EPOCHS}epochs.keras"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)


class CharacterTable:
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode a string C into a numpy array.
        # Arguments
            C: string, input processing.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            if c in self.char_indices:
                x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[x] for x in x)


# All the numbers, plus sign and space for padding.
chars = "0123456789+ "
ctable = CharacterTable(chars)

if os.path.exists(MODEL_PATH):
    print("Loading existing model.")
    model = keras.models.load_model(MODEL_PATH)
else:
    questions = []
    expected = []
    seen = set()
    print("Generating data...")
    while len(questions) < TRAINING_SIZE:
        f = lambda: int(
            "".join(
                np.random.choice(list("0123456789"))
                for i in range(np.random.randint(1, DIGITS + 1))
            )
        )
        a, b = f(), f()
        # Skip any addition questions we've already seen
        # Also skip any such that x+Y == Y+x (hence the sorting).
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        # Pad the data with spaces such that it is always MAXLEN.
        q = "{}+{}".format(a, b)
        query = q + " " * (MAXLEN - len(q))
        ans = str(a + b)
        # Answers can be of maximum size DIGITS + 1.
        ans += " " * (DIGITS + 1 - len(ans))
        if REVERSE:
            # Reverse the query, e.g., '12+345 ' becomes ' 543+21'. (Note the
            # space used for padding.)
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
    print("Total questions:", len(questions))


    print("Vectorization...")
    x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=bool)
    y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, DIGITS + 1)

    # Shuffle (x, y) in unison as the later parts of x will almost all be larger
    # digits.
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    print("Training Data:")
    print(x_train.shape)
    print(y_train.shape)

    print("Validation Data:")
    print(x_val.shape)
    print(y_val.shape)


    print("Build model...")
    num_layers = 1  # Try to add more LSTM layers!

    model = keras.Sequential()
    # "Encode" the input sequence using a LSTM, producing an output of size 128.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, num_feature).
    model.add(layers.Input((MAXLEN, len(chars))))
    model.add(layers.LSTM(128))
    # As the decoder RNN's input, repeatedly provide with the last output of
    # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
    # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
    model.add(layers.RepeatVector(DIGITS + 1))
    # The decoder RNN could be multiple layers stacked or a single layer.
    for _ in range(num_layers):
        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below expects
        # the first dimension to be the timesteps.
        model.add(layers.LSTM(128, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    model.add(layers.Dense(len(chars), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    class ETACallback(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.epoch_times = []

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            avg_time_per_epoch = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = self.params['epochs'] - (epoch + 1)
            eta_seconds = avg_time_per_epoch * remaining_epochs
            
            mins, secs = divmod(int(eta_seconds), 60)
            hours, mins = divmod(mins, 60)
            eta_str = f"{hours}h {mins}m {secs}s" if hours > 0 else f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            print(f" - Overall Training ETA: {eta_str}")

    # Train the model each generation and show predictions against the validation
    # dataset.
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[ETACallback()]
    )
    
    print(f"Saving model to {MODEL_PATH}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)

print("\n\nLaunching GUI...")
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog, messagebox

def on_model_select(event=None):
    global model, MAXLEN
    selected_file = model_var.get()
    path = os.path.join(MODELS_DIR, selected_file)
    print(f"Loading model {selected_file}...")
    model = keras.models.load_model(path)
    # Dynamically update MAXLEN based on the loaded model's input shape
    MAXLEN = model.input_shape[1]
    result_label.config(text=f"Loaded {selected_file}", fg="blue")

def predict_equation(event=None):
    user_input = entry.get()
    if not user_input:
        return

    # Strip spaces so "1 + 1" becomes "1+1" (which the model expects)
    user_input = user_input.replace(" ", "")

    # Validate input length
    if len(user_input) > MAXLEN:
        result_label.config(text=f"Input too long! Max {MAXLEN} chars.", fg="red")
        return

    # Validate characters
    if any(c not in chars for c in user_input):
        result_label.config(text="Invalid chars! Only digits and '+'", fg="red")
        return

    # Pre-process the user input
    query = user_input + " " * (MAXLEN - len(user_input))
    if REVERSE:
        query = query[::-1]

    # Vectorize the user input
    x_test = np.zeros((1, MAXLEN, len(chars)), dtype=bool)
    x_test[0] = ctable.encode(query, MAXLEN)

    # Make a prediction
    preds = model.predict(x_test, verbose=0)
    
    # Decode the prediction
    guess = ctable.decode(preds[0], calc_argmax=True)
    guess_stripped = guess.strip()
    
    # Check correctness
    color = "black"
    try:
        a, b = user_input.split('+')
        expected = str(int(a) + int(b))
        color = "green" if guess_stripped == expected else "red"
    except Exception:
        pass # Fallback to black if input isn't a standard 'a+b' format
        
    result_label.config(text=f"Result: {guess_stripped}", fg=color)

def auto_test():
    num_tests = simpledialog.askinteger("Auto Test", "How many additions to test?", parent=root, minvalue=1)
    if not num_tests:
        return
        
    # Create a new window for the scrolling results
    test_window = tk.Toplevel(root)
    test_window.title("Auto Test Results")
    test_window.geometry("400x500")
    
    text_area = tk.Text(test_window, font=("Courier", 14), state=tk.NORMAL)
    scrollbar = ttk.Scrollbar(test_window, command=text_area.yview)
    text_area.configure(yscrollcommand=scrollbar.set)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    text_area.tag_config("correct", foreground="green")
    text_area.tag_config("wrong", foreground="red")
    text_area.tag_config("stats", foreground="blue", font=("Helvetica", 14, "bold"))

    # Dynamically figure out the max digits based on the loaded model's MAXLEN
    current_digits = (MAXLEN - 1) // 2
    questions = []
    expected = []
    raw_questions = []
    
    for _ in range(num_tests):
        f = lambda: int(
            "".join(
                np.random.choice(list("0123456789"))
                for i in range(np.random.randint(1, current_digits + 1))
            )
        )
        a, b = f(), f()
        q = "{}+{}".format(a, b)
        query = q + " " * (MAXLEN - len(q))
        ans = str(a + b)
        ans += " " * (current_digits + 1 - len(ans))
        if REVERSE:
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
        raw_questions.append(q)
        
    x_test = np.zeros((num_tests, MAXLEN, len(chars)), dtype=bool)
    for i, sentence in enumerate(questions):
        x_test[i] = ctable.encode(sentence, MAXLEN)
        
    preds = model.predict(x_test, verbose=0)
    
    correct = 0
    for i in range(num_tests):
        guess = ctable.decode(preds[i], calc_argmax=True)
        is_correct = guess == expected[i]
        
        if is_correct:
            correct += 1
            tag = "correct"
            mark = "☑"
        else:
            tag = "wrong"
            mark = "☒"
            
        display_text = f"{raw_questions[i]:>9} = {guess.strip():<5} {mark}\n"
        text_area.insert(tk.END, display_text, tag)
        
        # Update the UI periodically so it visually scrolls
        if i % 50 == 0:
            text_area.see(tk.END)
            text_area.update()
            
    accuracy = (correct / num_tests) * 100
    stats_text = f"\nTested {num_tests} equations.\nCorrect: {correct}\nIncorrect: {num_tests - correct}\nAccuracy: {accuracy:.2f}%\n"
    text_area.insert(tk.END, stats_text, "stats")
    text_area.see(tk.END)
    text_area.config(state=tk.DISABLED) # Make read-only when finished

# Set up the Tkinter desktop window
root = tk.Tk()
root.title("RNN Addition")
root.geometry("400x300")

tk.Label(root, text="Select Model:", font=("Helvetica", 12)).pack(pady=(10, 0))

os.makedirs(MODELS_DIR, exist_ok=True)
available_models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")]
available_models.sort(reverse=True)

model_var = tk.StringVar(value=MODEL_FILENAME if MODEL_FILENAME in available_models else available_models[0])
dropdown = ttk.Combobox(root, textvariable=model_var, values=available_models, state="readonly", width=35)
dropdown.pack(pady=5)
dropdown.bind("<<ComboboxSelected>>", on_model_select)

tk.Label(root, text="Enter an addition problem:", font=("Helvetica", 14)).pack(pady=(10, 5))

entry = tk.Entry(root, font=("Helvetica", 16), justify="center")
entry.pack(pady=5)
entry.bind('<Return>', predict_equation) # Allows hitting the Enter key to solve
entry.focus()

tk.Button(root, text="Calculate", font=("Helvetica", 12), command=predict_equation).pack(pady=5)
tk.Button(root, text="Auto Test", font=("Helvetica", 12), command=auto_test).pack(pady=5)

result_label = tk.Label(root, text="", font=("Helvetica", 16, "bold"))
result_label.pack(pady=10)

# Launch the app
root.mainloop()
