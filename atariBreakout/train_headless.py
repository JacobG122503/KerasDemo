import os
import sys
import time
import json
import glob
import argparse

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

# Checkpoints are locally produced by this project and include a Lambda layer.
keras.config.enable_unsafe_deserialization()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 32
max_steps_per_episode = 10000
max_episodes = 0  # 0 = run until solved

# Replay / update cadence
epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
max_memory_length = 100000
update_after_actions = 4
update_target_network = 10000

# Early stopping is disabled by default so training runs until solved.
ENABLE_EARLY_STOP = False
EARLY_STOP_PATIENCE = 200        # episodes without improvement
EARLY_STOP_MIN_EPISODES = 500    # don't early-stop before this episode
EARLY_STOP_MIN_REWARD = 5.0      # don't early-stop until reward exceeds this
CHECKPOINT_EVERY_EPISODES = 50   # save a checkpoint every N episodes

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
IS_SLURM = bool(os.environ.get("SLURM_JOB_ID"))

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
num_actions = 4


def make_env(render_mode=None):
    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(env)
    env = FrameStackObservation(env, 4)
    return env


# ---------------------------------------------------------------------------
# Q-Network (Deepmind architecture)
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


def save_model(model, name):
    path = os.path.join(MODELS_DIR, name)
    model.save(path)
    print(f"Model saved to {path}")


def save_training_state(episode, frame_count, running_reward, epsilon_val, best_reward):
    state = {
        "episode": episode,
        "frame_count": frame_count,
        "running_reward": running_reward,
        "epsilon": epsilon_val,
        "best_reward": best_reward,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(MODELS_DIR, "training_state.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_checkpoint(model, model_target):
    """Auto-detect and load the latest checkpoint. Returns saved state dict or None."""
    state_path = os.path.join(MODELS_DIR, "training_state.json")
    if not os.path.exists(state_path):
        return None

    with open(state_path) as f:
        state = json.load(f)

    # Prefer the latest episode checkpoint; fall back to dqn_best
    episode_checkpoints = glob.glob(os.path.join(MODELS_DIR, "dqn_episode_*.keras"))
    checkpoint_path = None
    if episode_checkpoints:
        def ep_num(p):
            try:
                return int(os.path.basename(p).replace("dqn_episode_", "").replace(".keras", ""))
            except ValueError:
                return -1
        checkpoint_path = max(episode_checkpoints, key=ep_num)
    else:
        best_path = os.path.join(MODELS_DIR, "dqn_best.keras")
        if os.path.exists(best_path):
            checkpoint_path = best_path

    if checkpoint_path is None:
        print("training_state.json found but no model checkpoint — starting fresh.")
        return None

    print(f"[Resume] Loading checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"[Resume] Episode: {state['episode']}  Frames: {state['frame_count']}  "
          f"Running reward: {state['running_reward']:.2f}  Epsilon: {state['epsilon']:.4f}")

    loaded = keras.models.load_model(checkpoint_path, safe_mode=False)
    model.set_weights(loaded.get_weights())
    model_target.set_weights(loaded.get_weights())
    return state


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    print(f"Job ID: {JOB_ID}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true",
                        help="Start training from scratch, ignoring any saved checkpoint")
    args, _ = parser.parse_known_args()

    env = make_env()

    model = create_q_model()
    model_target = create_q_model()

    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    loss_function = keras.losses.Huber()

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []

    running_reward = 0
    episode_count = 0
    frame_count = 0
    best_running_reward = -float("inf")
    episodes_without_improvement = 0

    global epsilon
    eps = epsilon

    # --- Resume from checkpoint if available ---
    if not args.no_resume:
        saved = load_checkpoint(model, model_target)
        if saved:
            episode_count = saved["episode"]
            frame_count = saved["frame_count"]
            running_reward = saved["running_reward"]
            eps = max(saved["epsilon"], epsilon_min)
            best_running_reward = saved["best_reward"]
            # Seed episode_reward_history so the rolling mean starts correctly
            episode_reward_history = [running_reward] * min(100, episode_count)
    else:
        print("[Resume] --no-resume flag set — starting fresh.")

    # TensorBoard writer
    tb_writer = tf.summary.create_file_writer(os.path.join(LOGS_DIR, JOB_ID))

    start_time = time.time()

    while True:
        observation, _ = env.reset()
        state = np.array(observation)
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            frame_count += 1

            # Epsilon-greedy exploration
            if frame_count < epsilon_random_frames or eps > np.random.rand(1)[0]:
                action = np.random.choice(num_actions)
            else:
                state_tensor = keras.ops.convert_to_tensor(state)
                state_tensor = keras.ops.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                action = keras.ops.argmax(action_probs[0]).numpy()

            # Decay epsilon
            eps -= epsilon_interval / epsilon_greedy_frames
            eps = max(eps, epsilon_min)

            # Step environment
            state_next, reward, done, _, _ = env.step(action)
            state_next = np.array(state_next)
            episode_reward += reward

            # Store in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Train every 4th frame once we have enough samples
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = keras.ops.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Target Q-values
                future_rewards = model_target.predict(state_next_sample, verbose=0)
                updated_q_values = rewards_sample + gamma * keras.ops.amax(
                    future_rewards, axis=1
                )
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                masks = keras.ops.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    q_values = model(state_sample)
                    q_action = keras.ops.sum(keras.ops.multiply(q_values, masks), axis=1)
                    loss = loss_function(updated_q_values, q_action)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Update target network
            if frame_count % update_target_network == 0:
                model_target.set_weights(model.get_weights())
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit replay buffer size
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        # Episode bookkeeping
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

        # TensorBoard logging
        with tb_writer.as_default():
            tf.summary.scalar("episode_reward", episode_reward, step=episode_count)
            tf.summary.scalar("running_reward", running_reward, step=episode_count)
            tf.summary.scalar("epsilon", eps, step=episode_count)
            tf.summary.scalar("frame_count", frame_count, step=episode_count)

        # Periodic checkpoint
        if episode_count % CHECKPOINT_EVERY_EPISODES == 0:
            save_model(model, f"dqn_episode_{episode_count}.keras")
            save_training_state(episode_count, frame_count, running_reward, eps, best_running_reward)
            elapsed = time.time() - start_time
            print(
                f"[Checkpoint] Episode {episode_count} | "
                f"Running reward: {running_reward:.2f} | "
                f"Best: {best_running_reward:.2f} | "
                f"Epsilon: {eps:.4f} | "
                f"Frames: {frame_count} | "
                f"Elapsed: {elapsed/3600:.1f}h"
            )

        # Track best reward and save best model
        if running_reward > best_running_reward:
            best_running_reward = running_reward
            episodes_without_improvement = 0
            # Only save "best" once the agent is actually scoring
            if running_reward >= 1.0:
                save_model(model, "dqn_best.keras")
        else:
            episodes_without_improvement += 1

        # Early stopping: only after enough training AND the agent has
        # reached a meaningful reward level, so we don't quit while
        # the agent is still in the random-exploration phase.
        if (
            ENABLE_EARLY_STOP
            and episodes_without_improvement >= EARLY_STOP_PATIENCE
            and episode_count >= EARLY_STOP_MIN_EPISODES
            and best_running_reward >= EARLY_STOP_MIN_REWARD
        ):
            print(
                f"Early stopping at episode {episode_count}. "
                f"No improvement for {EARLY_STOP_PATIENCE} episodes. "
                f"Best running reward: {best_running_reward:.2f}"
            )
            save_model(model, f"dqn_early_stop_ep{episode_count}.keras")
            break

        # Solved condition
        if running_reward > 40:
            print(f"Solved at episode {episode_count}!")
            save_model(model, f"dqn_solved_ep{episode_count}.keras")
            break

        # Max episodes limit
        if max_episodes > 0 and episode_count >= max_episodes:
            print(f"Stopped at episode {episode_count} (max_episodes reached).")
            save_model(model, f"dqn_max_ep{episode_count}.keras")
            break

    # Always save a final model
    save_model(model, "dqn_final.keras")
    save_training_state(episode_count, frame_count, running_reward, eps, best_running_reward)

    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
