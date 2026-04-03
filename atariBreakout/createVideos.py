#!/usr/bin/env python3
"""
Generate MP4 videos of every trained DQN model playing Atari Breakout.

Usage:
    python createVideos.py                 # render all models
    python createVideos.py --max-steps 5000
    python createVideos.py --fps 30
    python createVideos.py --all-models    # include old checkpoints
"""
import os
import sys
import argparse
import re
from typing import Dict, List, Optional, Tuple

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


def select_models_to_render(models, render_all=False):
    if render_all:
        return models

    selected = set()

    # Keep primary artifacts when present.
    for name in ("dqn_best.keras", "dqn_final.keras"):
        if name in models:
            selected.add(name)

    # Keep exactly 5 spread-out episode checkpoints, based on max episode.
    episode_pattern = re.compile(r"^dqn_episode_(\d+)\.keras$")
    episode_models = []
    for model_name in models:
        match = episode_pattern.match(model_name)
        if match:
            episode_models.append((int(match.group(1)), model_name))

    if episode_models:
        episode_models.sort(key=lambda x: x[0])
        max_episode = episode_models[-1][0]
        target_count = min(5, len(episode_models))
        targets = [max_episode * i / 5 for i in range(1, 6)]

        chosen_episode_names = set()
        for target in targets:
            closest = min(
                episode_models,
                key=lambda item: (abs(item[0] - target), -item[0]),
            )
            chosen_episode_names.add(closest[1])
            if len(chosen_episode_names) >= target_count:
                break

        # If sparse checkpoints caused duplicates, fill with newest remaining.
        if len(chosen_episode_names) < target_count:
            for _, model_name in sorted(episode_models, key=lambda x: x[0], reverse=True):
                chosen_episode_names.add(model_name)
                if len(chosen_episode_names) >= target_count:
                    break

        selected.update(chosen_episode_names)

    return sorted(selected)


def _episode_from_name(model_name: str) -> Optional[int]:
    patterns = [
        re.compile(r"^dqn_episode_(\d+)\.keras$"),
        re.compile(r"^dqn_early_stop_ep(\d+)\.keras$"),
        re.compile(r"^dqn_solved_ep(\d+)\.keras$"),
        re.compile(r"^dqn_max_ep(\d+)\.keras$"),
    ]
    for pattern in patterns:
        match = pattern.match(model_name)
        if match:
            return int(match.group(1))
    return None


def build_model_episode_map(models: List[str]) -> Dict[str, Optional[int]]:
    """
    Estimate episode numbers for models that do not encode episode in filename.

    For dqn_best/dqn_final, use nearest prior dqn_episode_<N>.keras by mtime.
    """
    mapping: Dict[str, Optional[int]] = {name: _episode_from_name(name) for name in models}

    episode_candidates: List[Tuple[float, int]] = []
    for name in models:
        ep = _episode_from_name(name)
        if ep is None:
            continue
        if not name.startswith("dqn_episode_"):
            continue
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            episode_candidates.append((os.path.getmtime(path), ep))

    episode_candidates.sort(key=lambda x: x[0])
    if not episode_candidates:
        return mapping

    for special_name in ("dqn_best.keras", "dqn_final.keras"):
        if special_name not in mapping:
            continue
        special_path = os.path.join(MODELS_DIR, special_name)
        if not os.path.exists(special_path):
            continue
        special_mtime = os.path.getmtime(special_path)

        eligible = [ep for ts, ep in episode_candidates if ts <= special_mtime + 1e-6]
        if eligible:
            mapping[special_name] = max(eligible)

    return mapping


def model_to_video_name(model_name, episode_num=None):
    stem = model_name.replace(".keras", "")
    if model_name == "dqn_best.keras" and episode_num is not None:
        return f"dqn_best_ep{episode_num}.mp4"
    if model_name == "dqn_final.keras" and episode_num is not None:
        return f"dqn_final_ep{episode_num}.mp4"
    return f"{stem}.mp4"


def remove_stale_videos(models, episode_map):
    expected = {model_to_video_name(model_name, episode_map.get(model_name)) for model_name in models}
    for entry in os.listdir(MP4_DIR):
        if entry.endswith(".mp4") and entry not in expected:
            stale_path = os.path.join(MP4_DIR, entry)
            os.remove(stale_path)
            print(f"Removed stale video: {stale_path}")


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

    # Explicitly remove existing output so each run produces a fresh file.
    if os.path.exists(output_path):
        os.remove(output_path)

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
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Render every .keras model (including old checkpoints)",
    )
    args = parser.parse_args()

    all_models = get_available_models()
    if not all_models:
        print(f"No .keras models found in {MODELS_DIR}")
        sys.exit(1)

    models = select_models_to_render(all_models, render_all=args.all_models)
    episode_map = build_model_episode_map(models)

    os.makedirs(MP4_DIR, exist_ok=True)
    remove_stale_videos(models, episode_map)
    if args.all_models:
        print(f"Found {len(models)} model(s) (all). Videos will be saved to {MP4_DIR}/\n")
    else:
        print(
            f"Found {len(all_models)} model(s); rendering {len(models)} selected model(s). "
            f"Videos will be saved to {MP4_DIR}/\n"
        )

    for i, model_name in enumerate(models, 1):
        episode_num = episode_map.get(model_name)
        if episode_num is not None:
            print(f"[{i}/{len(models)}] {model_name} (episode ~{episode_num})")
        else:
            print(f"[{i}/{len(models)}] {model_name}")
        model = load_model(model_name)

        print(f"  Running episode (max {args.max_steps} steps)...")
        frames, reward, steps = run_episode(model, max_steps=args.max_steps)
        print(f"  Episode done — Reward: {reward:.0f}, Steps: {steps}, Frames: {len(frames)}")

        if not frames:
            print("  No frames captured, skipping.")
            continue

        video_name = model_to_video_name(model_name, episode_num)
        output_path = os.path.join(MP4_DIR, video_name)
        save_mp4(frames, output_path, fps=args.fps)
        print(f"  Saved → {output_path}\n")

    print("Done!")


if __name__ == "__main__":
    main()
