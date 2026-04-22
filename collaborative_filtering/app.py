import os
import argparse
import glob
import random
import numpy as np
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
import tkinter as tk
from tkinter import ttk, messagebox

os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import layers
from keras import ops

@keras.saving.register_keras_serializable()
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = ops.sum(user_vector * movie_vector, axis=1, keepdims=True)
        x = dot_user_movie + user_bias + movie_bias
        return ops.nn.sigmoid(x)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.num_users,
            "num_movies": self.num_movies,
            "embedding_size": self.embedding_size,
        })
        return config

class RecommenderApp:
    def __init__(self, root, model_path=None):
        self.root = root
        self.root.title("Movie Recommender")
        self.root.geometry("1100x800")
        
        self.model = None
        self.df = None
        self.movie_df = None
        self.user2user_encoded = None
        self.movie2movie_encoded = None
        self.movie_encoded2movie = None
        
        self.setup_ui()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(script_dir, "models")
        self.available_models = []
        if os.path.exists(self.models_dir):
            self.available_models = glob.glob(os.path.join(self.models_dir, "*.keras"))
            
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
        
        self.next_btn = tk.Button(control_frame, text="Get Recommendations for Random User ➔", command=self.load_random_user, state=tk.DISABLED, font=("Helvetica", 14), bg="#007bff", fg="black")
        self.next_btn.pack(side=tk.RIGHT)
        
        content_frame = tk.Frame(self.root)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        left_frame = tk.LabelFrame(content_frame, text="Movies Highly Rated by User", font=("Helvetica", 14, "bold"), padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.history_listbox = tk.Listbox(left_frame, font=("Helvetica", 12))
        self.history_listbox.pack(fill=tk.BOTH, expand=True)
        
        right_frame = tk.LabelFrame(content_frame, text="Top 10 Recommendations", font=("Helvetica", 14, "bold"), padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        self.rec_listbox = tk.Listbox(right_frame, font=("Helvetica", 12), fg="green")
        self.rec_listbox.pack(fill=tk.BOTH, expand=True)
        
    def prompt_model_selection(self):
        if not self.available_models:
            messagebox.showerror("No Models", "No trained models found in the 'models' directory.\nPlease finish training a model first!")
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
            # Provide the custom object RecommenderNet
            self.model = keras.models.load_model(path, custom_objects={"RecommenderNet": RecommenderNet})
            
            self.status_var.set("Downloading/Loading MovieLens Dataset...")
            self.root.update()
            
            movielens_data_file_url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            movielens_zipped_file = keras.utils.get_file("ml-latest-small.zip", movielens_data_file_url, extract=False)
            keras_datasets_path = Path(movielens_zipped_file).parents[0]
            movielens_dir = keras_datasets_path / "ml-latest-small"

            if not movielens_dir.exists():
                with ZipFile(movielens_zipped_file, "r") as zip:
                    zip.extractall(path=keras_datasets_path)

            self.df = pd.read_csv(movielens_dir / "ratings.csv")
            self.movie_df = pd.read_csv(movielens_dir / "movies.csv")
            
            user_ids = self.df["userId"].unique().tolist()
            self.user2user_encoded = {x: i for i, x in enumerate(user_ids)}
            movie_ids = self.df["movieId"].unique().tolist()
            self.movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
            self.movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
            
            self.status_var.set("Ready! Click 'Get Recommendations' to start.")
            self.next_btn.config(state=tk.NORMAL)
            
            self.load_random_user()
            
        except Exception as e:
            messagebox.showerror("Error Loading", f"Failed to initialize the app:\n{e}")
            self.root.destroy()

    def load_random_user(self):
        if self.df is None or self.model is None:
            return
            
        self.status_var.set("Calculating Recommendations...")
        self.next_btn.config(state=tk.DISABLED)
        self.root.update()
        
        # Pick a random user
        user_id = self.df.userId.sample(1).iloc[0]
        movies_watched_by_user = self.df[self.df.userId == user_id]
        
        # Find movies not watched
        movies_not_watched = self.movie_df[~self.movie_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]
        movies_not_watched = list(set(movies_not_watched).intersection(set(self.movie2movie_encoded.keys())))
        
        movies_not_watched_encoded = [[self.movie2movie_encoded.get(x)] for x in movies_not_watched]
        user_encoder = self.user2user_encoded.get(user_id)
        
        # Create user-movie array for prediction
        user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched_encoded), movies_not_watched_encoded))
        
        ratings = self.model.predict(user_movie_array, verbose=0).flatten()
        top_ratings_indices = ratings.argsort()[-10:][::-1]
        recommended_movie_ids = [self.movie_encoded2movie.get(movies_not_watched_encoded[x][0]) for x in top_ratings_indices]
        
        # Clear lists
        self.history_listbox.delete(0, tk.END)
        self.rec_listbox.delete(0, tk.END)
        
        # Populate history
        top_movies_user = movies_watched_by_user.sort_values(by="rating", ascending=False).head(15).movieId.values
        movie_df_rows = self.movie_df[self.movie_df["movieId"].isin(top_movies_user)]
        for row in movie_df_rows.itertuples():
            self.history_listbox.insert(tk.END, f"{row.title} ({row.genres})")
            
        # Populate recommendations
        recommended_movies = self.movie_df[self.movie_df["movieId"].isin(recommended_movie_ids)]
        for row in recommended_movies.itertuples():
            self.rec_listbox.insert(tk.END, f"{row.title} ({row.genres})")
            
        self.status_var.set(f"Showing Recommendations for User {user_id}")
        self.next_btn.config(state=tk.NORMAL)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    root = tk.Tk()
    app = RecommenderApp(root, args.model_path)
    root.mainloop()

if __name__ == "__main__":
    main()
