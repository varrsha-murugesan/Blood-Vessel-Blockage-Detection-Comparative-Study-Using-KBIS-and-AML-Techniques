import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import zipfile
os.system("pip install gdown")
import gdown

# ===============================
# 🔹 STEP 1: Download dataset from Google Drive
# ===============================

file_id = "1IpWYt-9c5AhVW3V5YQcPRpQb-p9jHjlH"
download_url = f"https://drive.google.com/uc?id={file_id}"

# Output zip name
output = "dataset_split.zip"

# Download from drive if not already present
if not os.path.exists(output):
    print("📥 Downloading dataset from Google Drive...")
    gdown.download(drive_link, output, quiet=False)
else:
    print("✅ Dataset already downloaded.")

# Extract zip file
if not os.path.exists("dataset_split"):
    print("📂 Extracting dataset...")
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(".")
    print("✅ Extraction complete!")
else:
    print("✅ Dataset folder already exists.")

IMG_SIZE = 128   
BATCH_SIZE = 4
EPOCHS = 10


def load_image(path, mask=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def load_dataset(image_dir, mask_dir):
    images, masks = [], []
    for file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, file)
        mask_path = os.path.join(mask_dir, file)
        if os.path.exists(mask_path):
            images.append(load_image(img_path))
            masks.append(load_image(mask_path, mask=True))
    return np.array(images), np.array(masks)


train_images, train_masks = load_dataset("dataset_split/train/images", "dataset_split/train/masks")
val_images, val_masks = load_dataset("dataset_split/val/images", "dataset_split/val/masks")

print("Train:", train_images.shape, train_masks.shape)
print("Val:", val_images.shape, val_masks.shape)


def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(num_filters, 3, padding="same", activation="relu")(x)
    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = layers.Input(input_shape)
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)
    b1 = conv_block(p4, 512)
    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(d4)
    return keras.Model(inputs, outputs, name="UNet")

model = build_unet((IMG_SIZE, IMG_SIZE, 1))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()


history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)


model.save("unet_model.h5")
print("✅ Model saved as unet_model.h5")
