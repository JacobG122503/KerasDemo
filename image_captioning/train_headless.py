import os
import subprocess
import time

os.environ["KERAS_BACKEND"] = "tensorflow"

import re
import numpy as np

import tensorflow as tf
import keras
from keras import layers
from keras.applications import efficientnet_v2
from keras.layers import TextVectorization
from keras.callbacks import EarlyStopping, TensorBoard
import pickle

from model_definition import (
    TransformerEncoderBlock,
    PositionalEmbedding,
    TransformerDecoderBlock,
    custom_standardization,
    SEQ_LENGTH, VOCAB_SIZE, EMBED_DIM, FF_DIM
)

# Paths and constants
IMAGES_PATH = "Flicker8k_Dataset"
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 64
EPOCHS_PER_RUN = [10,30,60,100] # Example: [20, 30, 40] or just [100]
AUTOTUNE = tf.data.AUTOTUNE

# Preparing the dataset
def load_captions_data(filename):
    """Loads captions (text) data and maps them to corresponding images."""
    with open(filename) as f:
        lines = f.readlines()
        
    captions_mapping = {}
    text_data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        img_name, caption = line.split('	')
        img_name = img_name.split('#')[0]
        img_name = os.path.join(IMAGES_PATH, img_name.strip())

        if img_name.endswith("jpg"):
            caption = "<start> " + caption.strip() + " <end>"
            text_data.append(caption)

            if img_name in captions_mapping:
                captions_mapping[img_name].append(caption)
            else:
                captions_mapping[img_name] = [caption]

    return captions_mapping, text_data

def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the captioning dataset into train and validation sets."""
    if shuffle:
        keys = list(caption_data.keys())
        np.random.shuffle(keys)
        caption_data = {key: caption_data[key] for key in keys}

    split = int(len(caption_data) * train_size)
    train_data = dict(list(caption_data.items())[:split])
    valid_data = dict(list(caption_data.items())[split:])
    return train_data, valid_data

# Load the dataset
captions_mapping, text_data = load_captions_data("Flickr8k.token.txt")

# Split the dataset
train_data, valid_data = train_val_split(captions_mapping)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))

# Vectorizing the text data

vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)
vectorization.adapt(text_data)

# Data augmentation
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
    ]
)

# Building a tf.data.Dataset pipeline for training
def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def process_input(img_path, captions):
    return decode_and_resize(img_path), vectorization(captions)

def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset

# Create the datasets
train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))

# Building the model
def get_cnn_model():
    base_model = efficientnet_v2.EfficientNetV2B0(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model

class ImageCaptioningModel(keras.Model):
    def __init__(
        self, cnn_model, encoder, decoder, num_captions_per_image=5, image_aug=None
    ):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        accuracy *= mask
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, captions, cap_mask, training):
        encoder_output = self.encoder(img_embed, training=training)
        y_true = captions[:, 1:]
        mask = cap_mask[:, 1:]
        y_pred = self.decoder(
            captions[:, :-1], encoder_output, training=training, mask=mask
        )
        loss = self.calculate_loss(y_true, y_pred, mask)
        acc = self.calculate_accuracy(y_true, y_pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        batch_x, batch_y = batch_data
        cap_mask = tf.math.not_equal(batch_y, 0)
        img_embed = self.cnn_model(batch_x)

        # Process all captions at once
        batch_y_reshaped = tf.reshape(batch_y, [-1, SEQ_LENGTH])
        cap_mask_reshaped = tf.reshape(cap_mask, [-1, SEQ_LENGTH])

        # Repeat image embeddings for each caption
        img_embed_repeated = tf.repeat(
            img_embed, repeats=self.num_captions_per_image, axis=0
        )

        with tf.GradientTape() as tape:
            loss, acc = self._compute_caption_loss_and_acc(
                img_embed_repeated, batch_y_reshaped, cap_mask_reshaped, training=True
            )

        # Collect all trainable variables
        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )

        # Get the gradients
        grads = tape.gradient(loss, train_vars)

        # Apply the gradients
        self.optimizer.apply_gradients(zip(grads, train_vars))

        # Update the metrics
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        # Return the metrics
        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc_tracker.result(),
        }

    def test_step(self, batch_data):
        batch_x, batch_y = batch_data
        cap_mask = tf.math.not_equal(batch_y, 0)
        img_embed = self.cnn_model(batch_x)

        # Process all captions at once
        batch_y_reshaped = tf.reshape(batch_y, [-1, SEQ_LENGTH])
        cap_mask_reshaped = tf.reshape(cap_mask, [-1, SEQ_LENGTH])

        # Repeat image embeddings for each caption
        img_embed_repeated = tf.repeat(
            img_embed, repeats=self.num_captions_per_image, axis=0
        )

        loss, acc = self._compute_caption_loss_and_acc(
            img_embed_repeated, batch_y_reshaped, cap_mask_reshaped, training=False
        )

        # Update the metrics
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        # Return the metrics
        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc_tracker.result(),
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

class HourlyProgressCallback(keras.callbacks.Callback):
    def __init__(self, email_address, total_epochs):
        super().__init__()
        self.email_address = email_address
        self.total_epochs = total_epochs
        self.last_email_time = time.time()
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        current_time = time.time()
        if (current_time - self.last_email_time) >= 3600:
            # It's been an hour, send an email
            self.last_email_time = current_time
            
            # Estimate completion time
            if self.epoch_start_time is not None:
                elapsed_time = current_time - self.epoch_start_time
                avg_time_per_epoch = elapsed_time / (epoch + 1)
                remaining_epochs = self.total_epochs - (epoch + 1)
                estimated_remaining_time = avg_time_per_epoch * remaining_epochs
                
                email_subject = "Keras Training Hourly Update"
                email_body = f"Epoch {epoch + 1}/{self.total_epochs} complete.\n\n"
                email_body += f"Estimated time remaining: {estimated_remaining_time / 3600:.2f} hours.\n"
                
                subprocess.run([
                    "python", "send_email.py",
                    self.email_address,
                    email_subject,
                    email_body
                ])

for epochs in EPOCHS_PER_RUN:
    print(f"--- Training for {epochs} epochs ---")

    # Model training
    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation
    )

    # Define the loss function
    cross_entropy = keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction="none"
    )

    # EarlyStopping criteria
    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

    # TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir=f"logs/e{epochs}")

    # Learning Rate Scheduler for the optimizer
    class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, post_warmup_learning_rate, warmup_steps):
            super().__init__()
            self.post_warmup_learning_rate = post_warmup_learning_rate
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            global_step = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            warmup_rate = global_step / warmup_steps
            warmup_learning_rate = warmup_rate * self.post_warmup_learning_rate
            return tf.cond(
                global_step < warmup_steps,
                lambda: warmup_learning_rate,
                lambda: self.post_warmup_learning_rate,
            )

    # Create a learning rate schedule
    num_train_steps = len(train_dataset) * epochs
    num_warmup_steps = num_train_steps // 15
    lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)

    # Compile the model
    caption_model.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=cross_entropy)

    # Fit the model
    hourly_callback = HourlyProgressCallback("jacobgar@iastate.edu", epochs)
    caption_model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        callbacks=[early_stopping, tensorboard_callback, hourly_callback],
    )

    # Save the model
    print(f"--- Saving model for {epochs} epochs ---")
    model_save_path = f"models/e{epochs}"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    caption_model.save_weights(f"{model_save_path}/caption_model.weights.h5")
    cnn_model.save(f"{model_save_path}/cnn_model")
    encoder.save(f"{model_save_path}/encoder")
    decoder.save(f"{model_save_path}/decoder")

    # Save the TextVectorization layer
    text_vec_model = keras.Sequential([vectorization])
    text_vec_model.compile()
    text_vec_model.save(f"{model_save_path}/text_vectorization")

    print(f"--- Training and saving for {epochs} epochs complete ---")

    email_subject = f"Keras Training Complete for {epochs} epochs"
    email_body = f"The Keras image captioning model training for {epochs} epochs has successfully completed.\n\n"
    email_body += "You can now download the models from the cluster."
    
    subprocess.run([
        "python", "send_email.py",
        "jacobgar@iastate.edu",
        email_subject,
        email_body
    ])

print("All training runs complete.")
