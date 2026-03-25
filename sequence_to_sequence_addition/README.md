# Sequence-to-Sequence Addition RNN

This project contains a Keras implementation of a sequence-to-sequence Recurrent Neural Network (RNN) that learns to add two numbers represented as strings.

## How to run

### 1. Setup your environment

Navigate to the project's directory.
```bash
cd sequence_to_sequence_addition
```

It is recommended to use a virtual environment to manage dependencies.

For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies

Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

### 3. Train the model

To start the training process, run the `app.py` script:
```bash
python app.py
```

The script will perform the following actions:
- Generate a dataset of number addition problems.
- Build an LSTM-based sequence-to-sequence model.
- Train the model for 30 epochs.
- After each epoch, it will display 10 random predictions from the validation set to show the model's progress. Correct predictions are marked with a green checkmark (☑), and incorrect ones with a red cross (☒).

## Using the model

The `app.py` script is self-contained and demonstrates both training and prediction. The predictions on the validation set at the end of each epoch serve as an example of how the model is used.

If you would like to save the trained model and use it in a separate script for predictions on new inputs, let me know and I can help with that.
