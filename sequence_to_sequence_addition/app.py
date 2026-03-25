import keras
from keras import layers
import numpy as np
import os
import time

# Parameters for the model and dataset.
TRAINING_SIZE = 50000
REVERSE = True

BATCH_SIZE = 32

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Automatically grab the latest model if one exists
existing_models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")]
existing_models.sort(reverse=True)
MODEL_FILENAME = existing_models[0] if existing_models else ""
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME) if MODEL_FILENAME else ""


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

model = None
MAXLEN = 0
if MODEL_PATH and os.path.exists(MODEL_PATH):
    print("Loading existing model.")
    model = keras.models.load_model(MODEL_PATH)
    MAXLEN = model.input_shape[1]

print("\n\nLaunching GUI...")
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog, messagebox
import threading

def start_training():
    train_digits = simpledialog.askinteger("Train Model", "How many digits to train on?", parent=root, minvalue=1, initialvalue=6)
    if not train_digits: return
    
    train_epochs = simpledialog.askinteger("Train Model", "How many epochs max?", parent=root, minvalue=1, initialvalue=60)
    if not train_epochs: return
    
    train_window = tk.Toplevel(root)
    train_window.title("Training Model")
    train_window.geometry("500x180")
    
    status_var = tk.StringVar(value="Preparing to train...")
    tk.Label(train_window, textvariable=status_var, font=("Helvetica", 12, "bold")).pack(pady=(20, 5))
    
    progress_var = tk.DoubleVar(value=0)
    progress_bar = ttk.Progressbar(train_window, variable=progress_var, maximum=train_epochs, length=400)
    progress_bar.pack(pady=10)
    
    metrics_var = tk.StringVar(value="")
    tk.Label(train_window, textvariable=metrics_var, font=("Courier", 10)).pack(pady=5)
    
    def update_status(msg):
        root.after(0, lambda: status_var.set(msg))
        
    def update_metrics(epoch, metrics_text):
        def apply_updates():
            progress_var.set(epoch)
            metrics_var.set(metrics_text)
        root.after(0, apply_updates)
        
    def run_training():
        update_status(f"Generating data for {train_digits}-digit addition...")
        local_maxlen = train_digits + 1 + train_digits
        questions = []
        expected = []
        seen = set()
        while len(questions) < TRAINING_SIZE:
            f = lambda: int("".join(np.random.choice(list("0123456789")) for i in range(np.random.randint(1, train_digits + 1))))
            a, b = f(), f()
            key = tuple(sorted((a, b)))
            if key in seen: continue
            seen.add(key)
            q = "{}+{}".format(a, b)
            query = q + " " * (local_maxlen - len(q))
            ans = str(a + b)
            ans += " " * (train_digits + 1 - len(ans))
            if REVERSE: query = query[::-1]
            questions.append(query)
            expected.append(ans)
        
        update_status("Vectorizing data...")
        x = np.zeros((len(questions), local_maxlen, len(chars)), dtype=bool)
        y = np.zeros((len(questions), train_digits + 1, len(chars)), dtype=bool)
        for i, sentence in enumerate(questions):
            x[i] = ctable.encode(sentence, local_maxlen)
        for i, sentence in enumerate(expected):
            y[i] = ctable.encode(sentence, train_digits + 1)
        
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        
        split_at = len(x) - len(x) // 10
        (x_train, x_val) = x[:split_at], x[split_at:]
        (y_train, y_val) = y[:split_at], y[split_at:]
        
        update_status("Building model...")
        new_model = keras.Sequential()
        new_model.add(layers.Input((local_maxlen, len(chars))))
        new_model.add(layers.LSTM(128))
        new_model.add(layers.RepeatVector(train_digits + 1))
        new_model.add(layers.LSTM(128, return_sequences=True))
        new_model.add(layers.Dense(len(chars), activation="softmax"))
        new_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        class GUICallback(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.epoch_times = []
                update_status("Training in progress...")
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
            def on_epoch_end(self, epoch, logs=None):
                epoch_time = time.time() - self.epoch_start_time
                self.epoch_times.append(epoch_time)
                avg = sum(self.epoch_times) / len(self.epoch_times)
                rem = self.params['epochs'] - (epoch + 1)
                mins, secs = divmod(int(avg * rem), 60)
                hrs, mins = divmod(mins, 60)
                eta_str = f"{hrs}h {mins}m {secs}s" if hrs > 0 else f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                
                m_text = f"Epoch {epoch+1}/{self.params['epochs']} | ETA: {eta_str}\nloss: {logs.get('loss',0):.4f} | acc: {logs.get('accuracy',0):.4f} | val_loss: {logs.get('val_loss',0):.4f} | val_acc: {logs.get('val_accuracy',0):.4f}"
                update_metrics(epoch + 1, m_text)

        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        
        history = new_model.fit(
            x_train, y_train, batch_size=BATCH_SIZE, epochs=train_epochs,
            validation_data=(x_val, y_val), callbacks=[GUICallback(), early_stopping], verbose=0
        )
        
        actual_epochs = len(history.epoch)
        filename = f"model_{train_digits}digits_{actual_epochs}epochs.keras"
        path = os.path.join(MODELS_DIR, filename)
        new_model.save(path)
        update_status(f"Training complete! Saved as {filename}")
        
        def update_main_ui():
            available_models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")]
            available_models.sort(reverse=True)
            dropdown['values'] = available_models
            model_var.set(filename)
            on_model_select()
            tk.Button(train_window, text="Close", command=train_window.destroy).pack(pady=5)
            
        root.after(0, update_main_ui) # Safely update GUI from the training thread

    threading.Thread(target=run_training, daemon=True).start()

def on_model_select(event=None):
    global model, MAXLEN
    selected_file = model_var.get()
    if not selected_file:
        return
    path = os.path.join(MODELS_DIR, selected_file)
    print(f"Loading model {selected_file}...")
    model = keras.models.load_model(path)
    # Dynamically update MAXLEN based on the loaded model's input shape
    MAXLEN = model.input_shape[1]
    result_label.config(text=f"Loaded {selected_file}", fg="blue")

def predict_equation(event=None):
    if not model:
        messagebox.showwarning("No Model", "Please select or train a model first.", parent=root)
        return
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
    if not model:
        messagebox.showwarning("No Model", "Please select or train a model first.", parent=root)
        return
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
root.geometry("400x340")

tk.Label(root, text="Select Model:", font=("Helvetica", 12)).pack(pady=(10, 0))

os.makedirs(MODELS_DIR, exist_ok=True)
available_models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")]
available_models.sort(reverse=True)

model_var = tk.StringVar(value=available_models[0] if available_models else "")
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
tk.Button(root, text="Train New Model", font=("Helvetica", 12), command=start_training).pack(pady=5)

result_label = tk.Label(root, text="", font=("Helvetica", 16, "bold"))
result_label.pack(pady=10)

# Launch the app
root.mainloop()
