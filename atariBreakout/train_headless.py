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
batch_size = 128
max_steps_per_episode = 10000
max_episodes = 0  # 0 = run until solved
NUM_ENVS = 16  # Parallel environments across cluster CPUs

# Replay / update cadence
epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
max_memory_length = 500000
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


def make_vec_env(num_envs):
    """Create N parallel async environments for faster data collection."""
    return gym.vector.AsyncVectorEnv(
        [lambda: make_env() for _ in range(num_envs)]
    )


# ---------------------------------------------------------------------------
# Replay Buffer — O(1) ring buffer backed by pre-allocated numpy arrays.
# Eliminates the catastrophic O(n) cost of `del list[:1]` on Python lists
# and enables fast vectorised sampling via fancy indexing.
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, state_shape=(4, 84, 84)):
        self.capacity = capacity
        self.index = 0
        self.size = 0

        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = float(done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states, actions, rewards, next_states, dones):
        """Insert N transitions at once — avoids per-env Python loop."""
        n = len(states)
        if self.index + n <= self.capacity:
            sl = slice(self.index, self.index + n)
            self.states[sl] = states
            self.actions[sl] = actions
            self.rewards[sl] = rewards
            self.next_states[sl] = next_states
            self.dones[sl] = dones
        else:
            # Wrap around the ring
            first = self.capacity - self.index
            self.states[self.index:] = states[:first]
            self.actions[self.index:] = actions[:first]
            self.rewards[self.index:] = rewards[:first]
            self.next_states[self.index:] = next_states[:first]
            self.dones[self.index:] = dones[:first]
            second = n - first
            self.states[:second] = states[first:]
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.next_states[:second] = next_states[first:]
            self.dones[:second] = dones[first:]
        self.index = (self.index + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size


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
        "episode": int(episode),
        "frame_count": int(frame_count),
        "running_reward": float(running_reward),
        "epsilon": float(epsilon_val),
        "best_reward": float(best_reward),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(MODELS_DIR, "training_state.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_checkpoint(model, model_target):
    """Load the best checkpoint (dqn_best.keras). Returns saved state dict or None."""
    state_path = os.path.join(MODELS_DIR, "training_state.json")
    if not os.path.exists(state_path):
        return None

    with open(state_path) as f:
        state = json.load(f)

    # Prefer dqn_best.keras, then fall back to the latest episode checkpoint.
    best_path = os.path.join(MODELS_DIR, "dqn_best.keras")
    if os.path.exists(best_path):
        checkpoint_path = best_path
        print(f"[Resume] Loading dqn_best.keras (best reward: {state['best_reward']:.2f})")
    else:
        # Find the latest dqn_episode_*.keras checkpoint
        episode_ckpts = sorted(glob.glob(os.path.join(MODELS_DIR, "dqn_episode_*.keras")))
        if episode_ckpts:
            checkpoint_path = episode_ckpts[-1]
            print(f"[Resume] dqn_best.keras not found — loading {os.path.basename(checkpoint_path)}")
        else:
            print("[Resume] No checkpoint found — starting fresh.")
            return None

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

    env = make_vec_env(NUM_ENVS)
    print(f"Running {NUM_ENVS} parallel environments.")

    model = create_q_model()
    model_target = create_q_model()

    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    loss_function = keras.losses.Huber()

    # Pre-allocated replay buffer (O(1) insert and sample)
    replay_buffer = ReplayBuffer(max_memory_length)
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
            # Always re-open exploration at the start of each job so the agent
            # can discover new strategies (e.g. tunneling). Epsilon will decay
            # back to epsilon_min naturally within ~220k frames.
            eps = 0.3
            best_running_reward = saved["best_reward"]
            # Seed episode_reward_history so the rolling mean starts correctly
            episode_reward_history = [running_reward] * min(100, episode_count)
    else:
        print("[Resume] --no-resume flag set — starting fresh.")

    # TensorBoard writer
    tb_writer = tf.summary.create_file_writer(os.path.join(LOGS_DIR, JOB_ID))

    start_time = time.time()

    # --- Compiled functions for maximum GPU throughput ---
    @tf.function(jit_compile=True)
    def select_actions(states_f32):
        """Batch Q-value inference for all envs — compiled to a single GPU kernel."""
        q_vals = model(states_f32, training=False)
        return tf.argmax(q_vals, axis=1, output_type=tf.int32)

    @tf.function
    def train_step(state_batch, next_state_batch, action_batch, reward_batch, done_batch):
        """Full Double-DQN training step compiled as a TF graph."""
        online_next_q = model(next_state_batch, training=False)
        best_next_actions = tf.argmax(online_next_q, axis=1)
        target_next_q = model_target(next_state_batch, training=False)
        best_masks = tf.one_hot(best_next_actions, num_actions)
        best_future = tf.reduce_sum(
            tf.cast(target_next_q, tf.float32) * best_masks, axis=1
        )
        updated_q = reward_batch + gamma * best_future
        updated_q = updated_q * (1.0 - done_batch) - done_batch

        action_masks = tf.one_hot(action_batch, num_actions)
        with tf.GradientTape() as tape:
            q_values = model(state_batch, training=True)
            q_action = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = loss_function(updated_q, q_action)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # --- Vectorized episode loop across NUM_ENVS parallel environments ---
    observations, infos = env.reset()
    states = np.array(observations)                       # (NUM_ENVS, 4, 84, 84)

    # Initial FIRE to launch balls in all envs
    fire_obs, _, _, _, _ = env.step(np.ones(NUM_ENVS, dtype=int))
    states = np.array(fire_obs)

    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    needs_fire = np.zeros(NUM_ENVS, dtype=bool)  # Track envs that just reset

    while True:
        frame_count += NUM_ENVS

        # Override actions for envs that just auto-reset: press FIRE to launch ball
        # instead of wasting frames with the ball sitting idle.
        # Epsilon-greedy: batch inference for all envs at once
        if frame_count < epsilon_random_frames or eps > np.random.rand():
            actions = np.random.randint(0, num_actions, size=NUM_ENVS)
        else:
            states_f32 = states.astype(np.float32)
            actions = select_actions(states_f32).numpy()

        # Force FIRE for envs that just reset
        actions[needs_fire] = 1
        needs_fire[:] = False

        eps -= (epsilon_interval / epsilon_greedy_frames) * NUM_ENVS
        eps = max(eps, epsilon_min)

        next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
        dones = terminateds | truncateds
        next_states = np.array(next_observations)

        # Batch-insert all transitions at once (no per-env Python loop)
        replay_buffer.add_batch(
            states, actions, rewards.astype(np.float32),
            next_states, dones.astype(np.float32)
        )
        episode_rewards += rewards

        states = next_states

        # Handle finished episodes
        for i in range(NUM_ENVS):
            if dones[i]:
                needs_fire[i] = True  # Will FIRE next step to re-launch ball
                episode_reward_history.append(episode_rewards[i])
                if len(episode_reward_history) > 100:
                    del episode_reward_history[:1]
                running_reward = np.mean(episode_reward_history)
                episode_count += 1
                episode_rewards[i] = 0.0

                # TensorBoard logging
                with tb_writer.as_default():
                    tf.summary.scalar("episode_reward", episode_reward_history[-1], step=episode_count)
                    tf.summary.scalar("running_reward", running_reward, step=episode_count)
                    tf.summary.scalar("epsilon", eps, step=episode_count)
                    tf.summary.scalar("frame_count", frame_count, step=episode_count)

                # Periodic checkpoint
                if episode_count % CHECKPOINT_EVERY_EPISODES == 0:
                    save_model(model, f"dqn_episode_{episode_count}.keras")
                    save_training_state(episode_count, frame_count, running_reward, eps, best_running_reward)
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(
                        f"[Checkpoint] Episode {episode_count} | "
                        f"Running reward: {running_reward:.2f} | "
                        f"Best: {best_running_reward:.2f} | "
                        f"Epsilon: {eps:.4f} | "
                        f"Frames: {frame_count} | "
                        f"FPS: {fps:.0f} | "
                        f"Elapsed: {elapsed/3600:.1f}h"
                    )

                # Track best reward and save best model
                if running_reward > best_running_reward:
                    best_running_reward = running_reward
                    episodes_without_improvement = 0
                    if running_reward >= 1.0:
                        save_model(model, "dqn_best.keras")
                else:
                    episodes_without_improvement += 1

                # Early stopping
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
                    save_model(model, "dqn_final.keras")
                    save_training_state(episode_count, frame_count, running_reward, eps, best_running_reward)
                    env.close()
                    print("Training complete.")
                    return

                # Solved condition
                if running_reward > 400:
                    print(f"Solved at episode {episode_count}!")
                    save_model(model, f"dqn_solved_ep{episode_count}.keras")
                    save_model(model, "dqn_final.keras")
                    save_training_state(episode_count, frame_count, running_reward, eps, best_running_reward)
                    env.close()
                    print("Training complete.")
                    return

                # Max episodes limit
                if max_episodes > 0 and episode_count >= max_episodes:
                    print(f"Stopped at episode {episode_count} (max_episodes reached).")
                    save_model(model, f"dqn_max_ep{episode_count}.keras")
                    save_model(model, "dqn_final.keras")
                    save_training_state(episode_count, frame_count, running_reward, eps, best_running_reward)
                    env.close()
                    print("Training complete.")
                    return

        # Train every update_after_actions frames (compiled graph — no Python overhead)
        if frame_count % update_after_actions == 0 and len(replay_buffer) > batch_size:
            s, a, r, s2, d = replay_buffer.sample(batch_size)
            train_step(
                tf.constant(s.astype(np.float32)),
                tf.constant(s2.astype(np.float32)),
                tf.constant(a),
                tf.constant(r),
                tf.constant(d),
            )

        # Update target network
        if frame_count % update_target_network == 0:
            model_target.set_weights(model.get_weights())
            print(f"running reward: {running_reward:.2f} at episode {episode_count}, frame count {frame_count}")


if __name__ == "__main__":
    main()
