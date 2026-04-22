import os
import argparse
import glob

# Set backend to JAX for optimal performance (optional, but recommended by keras)
os.environ["KERAS_BACKEND"] = "jax"

import keras
import keras_hub

def main():
    parser = argparse.ArgumentParser(description="Run inference with the trained BART summarization model.")
    parser.add_argument("--model_path", type=str, help="Path to the trained .keras model file")
    parser.add_argument("--text", type=str, help="The dialogue or text you want to summarize.")
    args = parser.parse_args()

    model_path = args.model_path

    if not model_path:
        # Find available models
        models_dir = "models"
        available_models = []
        if os.path.exists(models_dir):
            available_models = glob.glob(os.path.join(models_dir, "*.keras"))
        
        if not available_models:
            print("No trained models found in the 'models' directory.")
            print("Please finish training a model first, or specify the path with --model_path.")
            return

        print("Available models:")
        for i, model_file in enumerate(available_models):
            print(f"[{i+1}] {model_file}")
        
        while True:
            try:
                choice = input("\nEnter the number of the model you want to load (or 'q' to quit): ")
                if choice.lower() == 'q':
                    return
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_models):
                    model_path = available_models[choice_idx]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'")
        return

    print(f"Loading model from {model_path}...")
    # Load the entire saved model
    try:
        model = keras.models.load_model(model_path)
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
