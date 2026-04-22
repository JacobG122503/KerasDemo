import os
import argparse
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Multimodal Entailment Training")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
args = parser.parse_args()

os.environ["KERAS_BACKEND"] = "jax"
import keras
import keras_hub
from keras.utils import PyDataset

label_map = {"Contradictory": 0, "Implies": 1, "NoEntailment": 2}

print("Downloading dataset...")
image_base_path = keras.utils.get_file(
    "tweet_images",
    "https://github.com/sayakpaul/Multimodal-Entailment-Baseline/releases/download/v1.0.0/tweet_images.tar.gz",
    untar=True,
)

# The tar.gz file extracts its contents into a subfolder of the same name.
image_base_path = os.path.join(image_base_path, "tweet_images")

df = pd.read_csv("https://github.com/sayakpaul/Multimodal-Entailment-Baseline/raw/main/csvs/tweets.csv").iloc[0:1000]

print("Filtering missing images...")
valid_rows = []
for idx in range(len(df)):
    current_row = df.iloc[idx]
    id_1 = current_row["id_1"]
    id_2 = current_row["id_2"]
    ext_1 = current_row["image_1"].split(".")[-1]
    ext_2 = current_row["image_2"].split(".")[-1]
    img1_path = os.path.join(image_base_path, str(id_1) + f".{ext_1}")
    img2_path = os.path.join(image_base_path, str(id_2) + f".{ext_2}")
    
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        valid_rows.append({**current_row.to_dict(), "image_1_path": img1_path, "image_2_path": img2_path})

if not valid_rows:
    print("CRITICAL ERROR: No valid image pairs were found in the extracted directory!")
    print(f"Checked in: {image_base_path}")
    import sys
    sys.exit(1)

df = pd.DataFrame(valid_rows)
df["label_idx"] = df["label"].apply(lambda x: label_map[x])

train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"].values, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.05, stratify=train_df["label"].values, random_state=42)

print(f"Total training examples: {len(train_df)}")
print(f"Total validation examples: {len(val_df)}")
print(f"Total test examples: {len(test_df)}")

print("Initializing preprocessor...")
text_preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset(
    "bert_base_en_uncased",
    sequence_length=128,
)

bert_input_features = ["padding_mask", "segment_ids", "token_ids"]
def preprocess_text(text_1, text_2):
    output = text_preprocessor([text_1, text_2])
    return {feature: keras.ops.reshape(output[feature], [-1]) for feature in bert_input_features}

class UnifiedPyDataset(PyDataset):
    def __init__(self, df, batch_size=32, workers=4, use_multiprocessing=False, max_queue_size=10, **kwargs):
        super().__init__(**kwargs)
        self.dataframe = df
        self.image_x_1 = self.dataframe["image_1_path"].values
        self.image_x_2 = self.dataframe["image_2_path"].values
        self.image_y = self.dataframe["label_idx"].values
        self.text_x_1 = self.dataframe["text_1"].values
        self.text_x_2 = self.dataframe["text_2"].values
        self.text_y = self.dataframe["label_idx"].values
        self.batch_size = batch_size
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size

    def __getitem__(self, index):
        low = index * self.batch_size
        high = min(low + self.batch_size, len(self.image_x_1))

        batch_image_x_1 = self.image_x_1[low:high]
        batch_image_x_2 = self.image_x_2[low:high]
        batch_text_x_1 = self.text_x_1[low:high]
        batch_text_x_2 = self.text_x_2[low:high]
        batch_y = self.image_y[low:high]

        def process_img(file_name):
            img = resize(imread(file_name), (128, 128))
            if img.shape[2] == 4:
                return np.array(Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")) / 255.0
            return img

        image_1 = np.array([process_img(f) for f in batch_image_x_1])
        image_2 = np.array([process_img(f) for f in batch_image_x_2])

        text = {
            key: np.array([d[key] for d in [preprocess_text(t1, t2) for t1, t2 in zip(batch_text_x_1, batch_text_x_2)]])
            for key in ["padding_mask", "token_ids", "segment_ids"]
        }

        return (
            {"image_1": image_1, "image_2": image_2, "padding_mask": text["padding_mask"], "segment_ids": text["segment_ids"], "token_ids": text["token_ids"]},
            np.array(batch_y),
        )

    def __len__(self):
        return math.ceil(len(self.dataframe) / self.batch_size)

train_ds = UnifiedPyDataset(train_df, batch_size=32)
validation_ds = UnifiedPyDataset(val_df, batch_size=32)
test_ds = UnifiedPyDataset(test_df, batch_size=32)

def project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate):
    projected_embeddings = keras.layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = keras.ops.nn.gelu(projected_embeddings)
        x = keras.layers.Dense(projection_dims)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = keras.layers.LayerNormalization()(x)
    return projected_embeddings

def create_vision_encoder(num_projection_layers, projection_dims, dropout_rate, trainable=False):
    resnet_v2 = keras.applications.ResNet50V2(include_top=False, weights="imagenet", pooling="avg")
    for layer in resnet_v2.layers: layer.trainable = trainable

    image_1 = keras.Input(shape=(128, 128, 3), name="image_1")
    image_2 = keras.Input(shape=(128, 128, 3), name="image_2")

    preprocessed_1 = keras.applications.resnet_v2.preprocess_input(image_1)
    preprocessed_2 = keras.applications.resnet_v2.preprocess_input(image_2)

    embeddings_1 = resnet_v2(preprocessed_1)
    embeddings_2 = resnet_v2(preprocessed_2)
    embeddings = keras.layers.Concatenate()([embeddings_1, embeddings_2])

    outputs = project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate)
    return keras.Model([image_1, image_2], outputs, name="vision_encoder")

def create_text_encoder(num_projection_layers, projection_dims, dropout_rate, trainable=False):
    bert = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
    bert.trainable = trainable

    inputs = {feature: keras.Input(shape=(256,), dtype="int32", name=feature) for feature in ["padding_mask", "segment_ids", "token_ids"]}
    embeddings = bert(inputs)["pooled_output"]
    outputs = project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate)
    return keras.Model(inputs, outputs, name="text_encoder")

def create_multimodal_model(num_projection_layers=1, projection_dims=256, dropout_rate=0.1, vision_trainable=False, text_trainable=False):
    image_1 = keras.Input(shape=(128, 128, 3), name="image_1")
    image_2 = keras.Input(shape=(128, 128, 3), name="image_2")

    text_inputs = {feature: keras.Input(shape=(256,), dtype="int32", name=feature) for feature in ["padding_mask", "segment_ids", "token_ids"]}
    
    vision_encoder = create_vision_encoder(num_projection_layers, projection_dims, dropout_rate, vision_trainable)
    text_encoder = create_text_encoder(num_projection_layers, projection_dims, dropout_rate, text_trainable)

    vision_projections = vision_encoder([image_1, image_2])
    text_projections = text_encoder(text_inputs)

    concatenated = keras.layers.Concatenate()([vision_projections, text_projections])
    outputs = keras.layers.Dense(3, activation="softmax")(concatenated)
    
    # Needs to match the PyDataset exact dict keys or list order
    return keras.Model(inputs={"image_1": image_1, "image_2": image_2, **text_inputs}, outputs=outputs)

print("Compiling model...")
multimodal_model = create_multimodal_model()
multimodal_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print(f"Training model for {args.epochs} epochs...")
multimodal_model.fit(train_ds, validation_data=validation_ds, epochs=args.epochs)

print("Evaluating...")
_, acc = multimodal_model.evaluate(test_ds)
print(f"Accuracy on the test set: {round(acc * 100, 2)}%.")

os.makedirs("models", exist_ok=True)
model_path = f"models/multimodal_entailment_{args.epochs}_epochs.keras"
multimodal_model.save(model_path)
print(f"Model saved to {model_path}")
