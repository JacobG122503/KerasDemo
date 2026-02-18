import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_URL = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "nerf_model.keras")
MODEL_META_PATH = os.path.join(MODEL_DIR, "nerf_model_meta.txt")

BATCH_SIZE = 4
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 10
EPOCHS = 5
AUTO = tf.data.AUTOTUNE



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


def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=True):
    t_vals = tf.linspace(near, far, num_samples)
    if rand:
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = tf.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    rays = ray_origins[..., None, :] + (ray_directions[..., None, :] * t_vals[..., None])
    rays_flat = tf.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return (rays_flat, t_vals)


def render_rgb_depth(model, rays_flat, t_vals, height, width, train=True):
    if rays_flat.shape.rank == 3:
        batch_size = tf.shape(rays_flat)[0]
        feature_dim = tf.shape(rays_flat)[-1]
        rays_flat = tf.reshape(rays_flat, [-1, feature_dim])
    else:
        batch_size = 1

    if train:
        predictions = model(rays_flat)
    else:
        predictions = model.predict(rays_flat, verbose=0)

    predictions = tf.reshape(
        predictions, shape=(batch_size, height, width, NUM_SAMPLES, 4)
    )

    rgb = tf.sigmoid(predictions[..., :3])
    sigma = tf.nn.relu(predictions[..., 3])

    delta = t_vals[..., 1:] - t_vals[..., :-1]
    delta = tf.concat(
        [
            delta,
            tf.broadcast_to([1e10], shape=(batch_size, height, width, 1)),
        ],
        axis=-1,
    )
    alpha = 1.0 - tf.exp(-sigma * delta)

    exp_term = 1.0 - alpha
    transmittance = tf.math.cumprod(exp_term + 1e-10, axis=-1, exclusive=True)
    weights = alpha * transmittance

    rgb_map = tf.reduce_sum(weights[..., None] * rgb, axis=-2)
    depth_map = tf.reduce_sum(weights * t_vals, axis=-1)
    return (rgb_map, depth_map)


def build_nerf_model(input_dim, num_layers=8):
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for i in range(num_layers):
        x = layers.Dense(64, activation="relu")(x)
        if i > 0 and i % 4 == 0:
            x = layers.Concatenate()([x, inputs])
    outputs = layers.Dense(4)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


class NeRF(keras.Model):
    def __init__(self, nerf_model):
        super().__init__()
        self.nerf_model = nerf_model
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, inputs):
        images, rays = inputs
        rays_flat, t_vals = rays
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]

        with tf.GradientTape() as tape:
            rgb, _ = render_rgb_depth(
                self.nerf_model, rays_flat, t_vals, height, width, train=True
            )
            loss = self.loss_fn(images, rgb)

        gradients = tape.gradient(loss, self.nerf_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.nerf_model.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    data_path = keras.utils.get_file(origin=DATA_URL)
    data = np.load(data_path)
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    num_images, height, width = images.shape[:3]

    def map_pose_to_rays(pose):
        ray_origins, ray_directions = get_rays(height, width, focal, pose)
        return render_flat_rays(ray_origins, ray_directions, 2.0, 6.0, NUM_SAMPLES, rand=True)

    split_index = int(num_images * 0.8)
    train_images = images[:split_index]
    train_poses = poses[:split_index]

    train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
    train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
    train_ray_ds = train_pose_ds.map(map_pose_to_rays, num_parallel_calls=AUTO)
    train_ds = tf.data.Dataset.zip((train_img_ds, train_ray_ds))
    train_ds = train_ds.shuffle(BATCH_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTO)

    input_dim = 3 + 2 * 3 * POS_ENCODE_DIMS
    nerf_model = build_nerf_model(input_dim=input_dim)
    model = NeRF(nerf_model)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss_fn=keras.losses.MeanSquaredError())

    model.fit(train_ds, epochs=EPOCHS)

    for name in os.listdir(MODEL_DIR):
        if name.lower().endswith(".keras"):
            candidate = os.path.join(MODEL_DIR, name)
            if os.path.abspath(candidate) != os.path.abspath(MODEL_PATH):
                os.remove(candidate)

    nerf_model.save(MODEL_PATH)
    with open(MODEL_META_PATH, "w", encoding="utf-8") as meta_file:
        meta_file.write(f"epochs={EPOCHS}\n")

    print("Saved model to:", MODEL_PATH)


if __name__ == "__main__":
    main()
