import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import gradio as gr
import time
import threading

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
num_actions = 4

APP_CSS = """
footer { display: none !important; }
"""


# ---------------------------------------------------------------------------
# Model creation (must match training architecture)
# ---------------------------------------------------------------------------
def create_q_model():
    return keras.Sequential(
        [
            layers.Lambda(
                lambda tensor: keras.ops.transpose(tensor, [0, 2, 3, 1]),
                output_shape=(84, 84, 4),
                input_shape=(4, 84, 84),
            ),
            layers.Conv2D(32, 8, strides=4, activation="relu"),
            layers.Conv2D(64, 4, strides=2, activation="relu"),
            layers.Conv2D(64, 3, strides=1, activation="relu"),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(num_actions, activation="linear"),
        ]
    )


# ---------------------------------------------------------------------------
# Model discovery and loading
# ---------------------------------------------------------------------------
MODEL_CACHE = {}


def get_available_models():
    if not os.path.exists(MODELS_DIR):
        return []
    models = [
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".keras") and os.path.isfile(os.path.join(MODELS_DIR, f))
    ]
    return sorted(models)


def load_model(model_name):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    model_path = os.path.join(MODELS_DIR, model_name)
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    MODEL_CACHE[model_name] = model
    return model


# ---------------------------------------------------------------------------
# Game runner — plays one episode and returns frames + stats
# ---------------------------------------------------------------------------
def play_episode(model_name, max_steps=3000):
    """Play a single episode using the selected model. Returns (frames, total_reward)."""
    available = get_available_models()
    if not available:
        return None, 0, "No models found in models/ directory."

    if model_name not in available:
        return None, 0, f"Model '{model_name}' not found."

    model = load_model(model_name)

    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)

    observation, _ = env.reset()
    state = np.array(observation)

    frames = []
    total_reward = 0
    step = 0

    while step < max_steps:
        # Capture raw RGB frame for display
        raw_frame = env.render()
        if raw_frame is not None:
            frames.append(raw_frame)

        # Model selects action (greedy, no exploration)
        state_tensor = keras.ops.convert_to_tensor(state)
        state_tensor = keras.ops.expand_dims(state_tensor, 0)
        q_values = model(state_tensor, training=False)
        action = keras.ops.argmax(q_values[0]).numpy()

        state_next, reward, done, _, _ = env.step(action)
        state_next = np.array(state_next)

        total_reward += reward
        state = state_next
        step += 1

        if done:
            break

    env.close()
    return frames, total_reward, f"Episode complete — Reward: {total_reward:.0f}, Steps: {step}"


def run_game(model_name):
    """Gradio callback: play one episode and return results."""
    frames, reward, status = play_episode(model_name)
    if frames is None or len(frames) == 0:
        return None, status

    # Sample frames to create a reasonable gallery (every Nth frame)
    max_gallery = 30
    if len(frames) > max_gallery:
        step = len(frames) // max_gallery
        sampled = frames[::step][:max_gallery]
    else:
        sampled = frames

    return sampled, status


def refresh_models():
    models = get_available_models()
    if not models:
        return gr.Dropdown(choices=["(no models found)"], value="(no models found)")
    return gr.Dropdown(choices=models, value=models[0])


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_app():
    available = get_available_models()
    default_model = available[0] if available else "(no models found)"

    with gr.Blocks(css=APP_CSS, title="Atari Breakout DQN") as demo:
        gr.Markdown("# Atari Breakout — Deep Q-Network Agent")
        gr.Markdown("Select a trained model and watch the agent play Breakout.")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=available if available else ["(no models found)"],
                value=default_model,
                label="Select Model",
                interactive=True,
            )
            refresh_btn = gr.Button("Refresh Models", size="sm")

        play_btn = gr.Button("Play Episode", variant="primary")

        status_text = gr.Textbox(label="Status", interactive=False)
        gallery = gr.Gallery(label="Game Frames", columns=6, height="auto")

        refresh_btn.click(fn=refresh_models, outputs=model_dropdown)
        play_btn.click(fn=run_game, inputs=model_dropdown, outputs=[gallery, status_text])

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
