#!/usr/bin/env python3
"""
Generate MP4 videos of every trained DQN model playing Atari Breakout.

Usage:
    python play.py                        # render all models
    python play.py --max-steps 5000       # override max episode length
    python play.py --fps 30               # set video frame rate
"""
import os
import sys
import argparse

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import cv2
import tensorflow as tf
import keras
from keras import layers
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
NUM_ACTIONS = 4
DISPLAY_SCALE = 3  # scale up the tiny Atari frames
MP4_DIR = os.path.join(BASE_DIR, "mp4")


# ---------------------------------------------------------------------------
# Model creation (must match training architecture)
# ---------------------------------------------------------------------------
def create_q_model():
    return keras.Sequential(
        [
            layers.Input(shape=(4, 84, 84)),
            layers.Lambda(
                lambda tensor: keras.ops.transpose(tensor, [0, 2, 3, 1]),
                output_shape=(84, 84, 4),
            ),
            layers.Conv2D(32, 8, strides=4, activation="relu"),
            layers.Conv2D(64, 4, strides=2, activation="relu"),
            layers.Conv2D(64, 3, strides=1, activation="relu"),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(NUM_ACTIONS, activation="linear"),
        ]
    )


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def get_available_models():
    if not os.path.exists(MODELS_DIR):
        return []
    return sorted(
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".keras") and os.path.isfile(os.path.join(MODELS_DIR, f))
    )


def load_model(model_name):
    path = os.path.join(MODELS_DIR, model_name)
    print(f"Loading model: {path}")
    model = create_q_model()
    model.load_weights(path)
    return model


# ---------------------------------------------------------------------------
# Run one full episode, returning every frame
# ---------------------------------------------------------------------------
def run_episode(model, max_steps=3000):
    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode="rgb_array")
    env = AtariPreprocessing(env)
    env = FrameStackObservation(env, 4)

    observation, _ = env.reset()
    state = np.array(observation)

    frames = []
    total_reward = 0.0

    for step in range(1, max_steps + 1):
        raw_frame = env.render()
        if raw_frame is not None:
            frames.append(raw_frame)

        state_tensor = keras.ops.convert_to_tensor(state)
        state_tensor = keras.ops.expand_dims(state_tensor, 0)
        q_values = model(state_tensor, training=False)
        action = keras.ops.argmax(q_values[0]).numpy()

        state_next, reward, done, _, _ = env.step(action)
        state = np.array(state_next)
        total_reward += reward

        if done:
            raw_frame = env.render()
            if raw_frame is not None:
                frames.append(raw_frame)
            break

    env.close()
    return frames, total_reward, step


# ---------------------------------------------------------------------------
# Write frames to MP4
# ---------------------------------------------------------------------------
def save_mp4(frames, output_path, fps=30):
    h, w = frames[0].shape[:2]
    out_w, out_h = w * DISPLAY_SCALE, h * DISPLAY_SCALE

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    for frame in frames:
        # Resize with nearest-neighbor to keep pixel art crisp
        big = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        # RGB → BGR for OpenCV
        writer.write(cv2.cvtColor(big, cv2.COLOR_RGB2BGR))

    writer.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate MP4 videos for all trained Atari Breakout DQN models"
    )
    parser.add_argument("--max-steps", type=int, default=3000, help="Max steps per episode")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    args = parser.parse_args()

    models = get_available_models()
    if not models:
        print(f"No .keras models found in {MODELS_DIR}")
        sys.exit(1)

    os.makedirs(MP4_DIR, exist_ok=True)
    print(f"Found {len(models)} model(s). Videos will be saved to {MP4_DIR}/\n")

    for i, model_name in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {model_name}")
        model = load_model(model_name)

        print(f"  Running episode (max {args.max_steps} steps)...")
        frames, reward, steps = run_episode(model, max_steps=args.max_steps)
        print(f"  Episode done — Reward: {reward:.0f}, Steps: {steps}, Frames: {len(frames)}")

        if not frames:
            print("  No frames captured, skipping.")
            continue

        video_name = model_name.replace(".keras", ".mp4")
        output_path = os.path.join(MP4_DIR, video_name)
        save_mp4(frames, output_path, fps=args.fps)
        print(f"  Saved → {output_path}\n")

    print("Done!")


if __name__ == "__main__":
    main()
