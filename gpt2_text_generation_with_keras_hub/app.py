import os
import argparse
import glob
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

os.environ["KERAS_BACKEND"] = "jax"
import keras
import keras_hub

class GPT2App:
    def __init__(self, root, model_path=None):
        self.root = root
        self.root.title("GPT-2 Text Generation (KerasHub)")
        self.root.geometry("800x600")
        
        self.model = None
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(script_dir, "models")
        self.available_models = []
        if os.path.exists(self.models_dir):
            self.available_models = glob.glob(os.path.join(self.models_dir, "*.keras"))
            
        self.setup_ui()
        
        if model_path and os.path.exists(model_path):
            self.root.after(100, lambda: self.load_model(model_path))
        else:
            self.root.after(100, self.prompt_model_selection)

    def setup_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing GUI...")
        tk.Label(control_frame, textvariable=self.status_var, font=("Helvetica", 14, "bold")).pack(side=tk.LEFT)
        
        input_frame = tk.Frame(self.root)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)
        
        tk.Label(input_frame, text="Prompt:", font=("Helvetica", 12)).pack(side=tk.LEFT)
        self.prompt_entry = tk.Entry(input_frame, font=("Helvetica", 12))
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.prompt_entry.insert(0, "I like basketball")
        
        self.gen_btn = tk.Button(input_frame, text="Generate", command=self.generate, state=tk.DISABLED, font=("Helvetica", 12), bg="#007bff")
        self.gen_btn.pack(side=tk.RIGHT)
        
        self.output_text = scrolledtext.ScrolledText(self.root, font=("Helvetica", 12), wrap=tk.WORD)
        self.output_text.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=20, pady=20)

    def prompt_model_selection(self):
        if not self.available_models:
            messagebox.showerror("No Models", "No trained models found.\nPlease finish training a model first!")
            self.root.destroy()
            return
            
        top = tk.Toplevel(self.root)
        top.title("Select Model")
        top.geometry("450x150")
        top.transient(self.root)
        top.grab_set()
        
        tk.Label(top, text="Select a trained Keras model to load:", font=("Helvetica", 12)).pack(pady=15)
        
        combo = ttk.Combobox(top, values=[os.path.basename(m) for m in self.available_models], state="readonly", width=45, font=("Helvetica", 12))
        combo.current(0)
        combo.pack(pady=5)
        
        def on_select():
            selected_model = self.available_models[combo.current()]
            top.destroy()
            self.root.after(50, lambda: self.load_model(selected_model))
            
        tk.Button(top, text="Load Model", command=on_select, font=("Helvetica", 12, "bold")).pack(pady=15)

    def load_model(self, path):
        self.status_var.set(f"Loading Model: {os.path.basename(path)}...")
        self.root.update()
        try:
            self.model = keras.models.load_model(path)
            self.status_var.set("Ready!")
            self.gen_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{e}")
            self.root.destroy()
            
    def generate(self):
        if not self.model:
            return
            
        self.status_var.set("Generating...")
        self.gen_btn.config(state=tk.DISABLED)
        self.root.update()
        
        prompt = self.prompt_entry.get()
        output = self.model.generate(prompt, max_length=200)
        
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, output)
        
        self.status_var.set("Ready!")
        self.gen_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    
    root = tk.Tk()
    app = GPT2App(root, args.model_path)
    root.mainloop()
