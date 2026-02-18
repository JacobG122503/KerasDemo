import os
import re
import sys
import numpy as np
import imageio.v2 as imageio
import tensorflow as tf
from tensorflow import keras

DATA_URL = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "nerf_model.keras")
MODEL_META_PATH = os.path.join(os.path.dirname(__file__), "models", "nerf_model_meta.txt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

NUM_SAMPLES = 32
POS_ENCODE_DIMS = 10
NUM_FRAMES = 120
FPS = 30


def load_epoch_label(model_path):
    if os.path.exists(MODEL_META_PATH):
        with open(MODEL_META_PATH, "r", encoding="utf-8") as meta_file:
            for line in meta_file:
                if line.startswith("epochs="):
                    value = line.split("=", 1)[1].strip()
                    if value.isdigit():
                        return f"{value}Epochs"

    base_name = os.path.basename(model_path)
    match = re.search(r"(\d+)\s*epochs?", base_name, re.IGNORECASE)
    if match:
        return f"{match.group(1)}Epochs"

    fallback = re.search(r"(\d+)", base_name)
    return f"{fallback.group(1)}Epochs" if fallback else "unknownEpochs"


def build_video_path(model_path):
    epoch_label = load_epoch_label(model_path)
    base_output = os.path.join(OUTPUT_DIR, f"rgb_video_{epoch_label}.mp4")
    if not os.path.exists(base_output):
        return base_output

    index = 2
    while True:
        candidate = os.path.join(OUTPUT_DIR, f"rgb_video_{epoch_label}_v{index}.mp4")
        if not os.path.exists(candidate):
            return candidate
        index += 1


def encode_position(x):
    positions = [x]
    for i in range(POS_ENCODE_DIMS):
        scale = 2.0 ** i
        positions.append(tf.sin(scale * x))
        positions.append(tf.cos(scale * x))
    return tf.concat(positions, axis=-1)


def get_rays(height, width, focal, pose):
    i, j = tf.meshgrid(
        tf.range(width, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        indexing="xy",
    )
    transformed_i = (i - width * 0.5) / focal
    transformed_j = (j - height * 0.5) / focal

    directions = tf.stack([transformed_i, -transformed_j, -tf.ones_like(i)], axis=-1)
    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]

    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = tf.reduce_sum(camera_dirs, axis=-1)
    ray_origins = tf.broadcast_to(height_width_focal, tf.shape(ray_directions))
    return (ray_origins, ray_directions)


def render_flat_rays(ray_origins, ray_directions, near, far, num_samples):
    t_vals = tf.linspace(near, far, num_samples)
    rays = ray_origins[..., None, :] + (ray_directions[..., None, :] * t_vals[..., None])
    rays_flat = tf.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return (rays_flat, t_vals)


def render_rgb(model, rays_flat, t_vals, height, width):
    predictions = model(rays_flat)
    predictions = tf.reshape(predictions, shape=(height, width, NUM_SAMPLES, 4))

    rgb = tf.sigmoid(predictions[..., :3])
    sigma = tf.nn.relu(predictions[..., 3])

    delta = t_vals[..., 1:] - t_vals[..., :-1]
    delta = tf.concat([delta, tf.broadcast_to([1e10], shape=(1,))], axis=-1)
    alpha = 1.0 - tf.exp(-sigma * delta)

    exp_term = 1.0 - alpha
    transmittance = tf.math.cumprod(exp_term + 1e-10, axis=-1, exclusive=True)
    weights = alpha * transmittance

    rgb_map = tf.reduce_sum(weights[..., None] * rgb, axis=-2)
    return rgb_map


def get_translation_t(t):
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def get_rotation_phi(phi):
    matrix = [
        [1, 0, 0, 0],
        [0, tf.cos(phi), -tf.sin(phi), 0],
        [0, tf.sin(phi), tf.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def get_rotation_theta(theta):
    matrix = [
        [tf.cos(theta), 0, -tf.sin(theta), 0],
        [0, 1, 0, 0],
        [tf.sin(theta), 0, tf.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def pose_spherical(theta, phi, t):
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found:", MODEL_PATH)
        print("Run train.py first to create a model file.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_path = keras.utils.get_file(origin=DATA_URL)
    data = np.load(data_path)
    images = data["images"]
    focal = data["focal"]

    height, width = images.shape[1:3]

    model = keras.models.load_model(MODEL_PATH)

    frames = []
    for theta in np.linspace(0.0, 360.0, NUM_FRAMES, endpoint=False):
        pose = pose_spherical(theta, -30.0, 4.0)
        ray_origins, ray_directions = get_rays(height, width, focal, pose)
        rays_flat, t_vals = render_flat_rays(ray_origins, ray_directions, 2.0, 6.0, NUM_SAMPLES)
        rgb = render_rgb(model, rays_flat, t_vals, height, width)
        rgb = tf.clip_by_value(rgb, 0.0, 1.0)
        rgb_uint8 = tf.cast(rgb * 255.0, tf.uint8).numpy()
        frames.append(rgb_uint8)

    video_path = build_video_path(MODEL_PATH)
    imageio.mimwrite(video_path, frames, fps=FPS, quality=7, macro_block_size=None)
    print("Saved video to:", video_path)


if __name__ == "__main__":
    main()
