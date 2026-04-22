import os
import argparse

# Set backend to JAX for optimal performance (optional, but recommended by keras)
os.environ["KERAS_BACKEND"] = "jax"

import keras
import keras_hub

def main():
    parser = argparse.ArgumentParser(description="Run inference with the trained BART summarization model.")
    parser.add_argument("--model_path", type=str, default="models/bart_summarization_10_epochs.keras", help="Path to the trained .keras model file")
    parser.add_argument("--text", type=str, help="The dialogue or text you want to summarize.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at '{args.model_path}'")
        print("Make sure you have finished training the model first!")
        return

    print(f"Loading model from {args.model_path}...")
    # Load the entire saved model
    try:
        model = keras.models.load_model(args.model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if args.text:
        print("\n--- Input Dialogue ---")
        print(args.text)
        print("\n--- Generating Summary ---")
        summary = model.generate(args.text, max_length=40)
        print(summary)
    else:
        print("\nModel loaded successfully! Enter a dialogue to summarize (or type 'quit' to exit):")
        while True:
            try:
                user_input = input("\nDialogue: ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if not user_input.strip():
                    continue
                
                print("\nSummary: ", end="")
                summary = model.generate(user_input, max_length=40)
                print(summary)
            except KeyboardInterrupt:
                break
            except EOFError:
                break

if __name__ == "__main__":
    main()
