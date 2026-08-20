"""
train_model.py - PC Training & TFLite Export Script
Trains a MobileNetV2 transfer learning model on skin condition images
and exports 'rash_model.tflite' and 'labels.txt' for Raspberry Pi 4 edge deployment.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from prepare_dataset import extract_and_prepare_dataset

def build_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Builds a transfer learning model using MobileNetV2 pre-trained on ImageNet.
    Base layers are frozen; custom classification head is appended.
    """
    # 1. Base Pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze pre-trained weights

    # 2. Data Augmentation Layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
    ], name="data_augmentation")

    # 3. Model Architecture
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="RashScanner_MobileNetV2")
    return model

def export_tflite(model, tflite_path="rash_model.tflite", quantize=True):
    """
    Converts Keras model to TensorFlow Lite (.tflite) format with optional quantization.
    """
    print(f"\n[Info] Converting model to TFLite format (Quantization={quantize})...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"[Success] Exported TFLite model: '{tflite_path}' ({size_mb:.2f} MB)")

def export_labels(class_names, labels_path="labels.txt"):
    """Saves detected class labels to a text file, one label per line."""
    with open(labels_path, 'w', encoding='utf-8') as f:
        for label in class_names:
            f.write(f"{label}\n")
    print(f"[Success] Exported class labels to '{labels_path}' ({len(class_names)} classes).")

def train_and_export(dataset_dir="dataset", img_size=224, batch_size=32, epochs=10, tflite_path="rash_model.tflite", labels_path="labels.txt"):
    """Main training, evaluation, and export pipeline."""
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print(f"[Notice] Dataset directory '{dataset_dir}' empty or missing. Attempting automatic preparation...")
        extract_and_prepare_dataset(target_dir=dataset_dir)

    print(f"\n[Info] Loading dataset from '{dataset_dir}'...")
    
    # Load Training Dataset (80%)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    # Load Validation Dataset (20%)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"[Dataset] Found {num_classes} classes: {class_names[:10]}...")

    # Performance Autotuning
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build and Compile Model
    print(f"\n[Model] Compiling MobileNetV2 with {num_classes} classification outputs...")
    model = build_model(input_shape=(img_size, img_size, 3), num_classes=num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]

    # Train Head
    print(f"\n[Training] Starting transfer learning training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Evaluate Model
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\n[Validation Result] Loss: {val_loss:.4f} | Accuracy: {val_acc*100:.2f}%")

    # Save Native Keras Model
    keras_path = "rash_model.h5"
    model.save(keras_path)
    print(f"[Success] Saved Keras model to '{keras_path}'.")

    # Export Labels & TFLite Model
    export_labels(class_names, labels_path=labels_path)
    export_tflite(model, tflite_path=tflite_path, quantize=True)
    print("\n[Complete] PC Model Training & Edge Export Pipeline Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Rash Classification Model & Export TFLite")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Directory containing class subfolders")
    parser.add_argument("--img-size", type=int, default=224, help="Input image dimension (224x224)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--tflite-path", type=str, default="rash_model.tflite", help="Target TFLite output path")
    parser.add_argument("--labels-path", type=str, default="labels.txt", help="Target labels text file path")
    
    args = parser.parse_args()
    
    train_and_export(
        dataset_dir=args.dataset_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        tflite_path=args.tflite_path,
        labels_path=args.labels_path
    )
