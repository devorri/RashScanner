"""
train_10_model.py - Train 10-Class Skin Disease Model for Real-Time Rash Scanner
Architecture: MobileNetV2 + Transfer Learning + Fine-Tuning + TFLite Quantized Export
10 Distinct Classes:
1. Acne_Vulgaris
2. Chickenpox_Varicella
3. Eczema
4. Hives
5. Impetigo
6. Melanoma
7. Psoriasis
8. Ringworm
9. Scabies
10. Warts
"""

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
FINE_TUNE_EPOCHS = 8

CLASSES = [
    "Acne_Vulgaris",
    "Chickenpox_Varicella",
    "Eczema",
    "Hives",
    "Impetigo",
    "Melanoma",
    "Psoriasis",
    "Ringworm",
    "Scabies",
    "Warts"
]

DATASET_ROOT = "dataset_boosted"
SUBSET_ROOT = "dataset_10_classes"
MODEL_TFLITE_PATH = "rash_model.tflite"
LABELS_PATH = "labels.txt"

def prepare_10_class_dataset():
    """Create a clean subset directory containing only the 10 target classes."""
    print(f"Preparing subset dataset directory: '{SUBSET_ROOT}'...")
    if os.path.exists(SUBSET_ROOT):
        shutil.rmtree(SUBSET_ROOT)
    os.makedirs(SUBSET_ROOT, exist_ok=True)

    for cls in CLASSES:
        src = os.path.join(DATASET_ROOT, cls)
        dst = os.path.join(SUBSET_ROOT, cls)
        if os.path.exists(src):
            shutil.copytree(src, dst)
            count = len(os.listdir(dst))
            print(f"  [+] {cls}: {count} images copied")
        else:
            print(f"  [!] Missing source folder: {src}")

    # Write labels.txt
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for cls in CLASSES:
            f.write(f"{cls}\n")
    print(f"Written {len(CLASSES)} classes to '{LABELS_PATH}'")

def train_and_export():
    prepare_10_class_dataset()

    print("\nLoading Training & Validation Datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        SUBSET_ROOT,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASSES
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        SUBSET_ROOT,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASSES
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data Augmentation Pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")

    # Base MobileNetV2
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    # Model Assembly
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n--- STAGE 1: Training Classification Head ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )

    # STAGE 2: Fine-Tuning Top Layers of MobileNetV2
    print("\n--- STAGE 2: Fine-Tuning MobileNetV2 Top Layers ---")
    base_model.trainable = True
    # Freeze the bottom 100 layers and fine-tune the rest
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[reduce_lr],
        verbose=1
    )

    # Convert to TFLite
    print(f"\nExporting model to TFLite format: '{MODEL_TFLITE_PATH}'...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(MODEL_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Model saved successfully to '{MODEL_TFLITE_PATH}' ({len(tflite_model) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    train_and_export()
