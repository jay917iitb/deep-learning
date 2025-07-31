import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Paths
# --- Make sure to update this path to your local directory ---
BASE_DIR = r"C:\Users\legen\OneDrive - Indian Institute of Technology Bombay\Documents\Coding\SOC-25_Intro-to-Deep-Learning-main\CUB_200_2011"
IMAGE_DIR = os.path.join(BASE_DIR, 'images')

# Load dataset
def load_data():
    images_txt = pd.read_csv(os.path.join(BASE_DIR, 'images.txt'), sep=' ', header=None, names=['img_id', 'filepath'])
    labels_txt = pd.read_csv(os.path.join(BASE_DIR, 'image_class_labels.txt'), sep=' ', header=None, names=['img_id', 'label'])
    split_txt = pd.read_csv(os.path.join(BASE_DIR, 'train_test_split.txt'), sep=' ', header=None, names=['img_id', 'is_train'])

    data = images_txt.merge(labels_txt, on='img_id').merge(split_txt, on='img_id')
    data['filepath'] = data['filepath'].apply(lambda x: os.path.join(IMAGE_DIR, x))
    data['label'] -= 1  # class labels: 0 to 199
    return data

data_df = load_data()
train_df = data_df[data_df['is_train'] == 1]
test_df = data_df[data_df['is_train'] == 0]

# Constants
IMG_SIZE = (224, 224)
# --- FIX: Reduced batch size to prevent OOM error during fine-tuning ---
BATCH_SIZE = 16 
NUM_CLASSES = 200
AUTOTUNE = tf.data.AUTOTUNE
INITIAL_EPOCHS = 10 # Epochs for the feature extraction phase
FINETUNE_EPOCHS = 15 # Epochs for the fine-tuning phase

# --- Updated Preprocessing Function ---
# One-hot encode the labels for CategoricalCrossentropy
def preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    
    # One-hot encode the label
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return image, label

def prepare_dataset(df, shuffle=False):
    paths = df['filepath'].values
    labels = df['label'].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = prepare_dataset(train_df, shuffle=True)
test_ds = prepare_dataset(test_df)

# --- Improved Model with Transfer Learning ---
def create_model():
    # 1. Data Augmentation Layers
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")

    # 2. Base Model (EfficientNetV2B0 pre-trained on ImageNet)
    base_model = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=IMG_SIZE + (3,)
    )
    # Freeze the base model initially
    base_model.trainable = False

    # 3. Model Construction
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    # Rescale inputs to [-1, 1] as expected by EfficientNet
    x = keras.applications.efficientnet_v2.preprocess_input(x)
    x = base_model(x, training=False) # Important for BN layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

model = create_model()
model.summary()

# --- Compile with CategoricalCrossentropy ---
# Using Adam optimizer as a fallback for older TensorFlow versions
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# --- Phase 1: Feature Extraction Training ---
print("--- Starting Feature Extraction Phase ---")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=INITIAL_EPOCHS,
    verbose=2
)

# --- Phase 2: Fine-Tuning ---
# Unfreeze the base model and use a very low learning rate
model.get_layer('efficientnetv2-b0').trainable = True
model.summary()

# --- Re-compile for Fine-Tuning ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Crucial to use a low LR
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

print("\n--- Starting Fine-Tuning Phase ---")
history_fine = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=INITIAL_EPOCHS + FINETUNE_EPOCHS,
    initial_epoch=history.epoch[-1] + 1, # Continue from where we left off
    callbacks=[keras.callbacks.ModelCheckpoint('fine_tuned_best_model.h5', save_best_only=True, monitor='val_accuracy')],
    verbose=2
)

# Evaluation
# Load the best weights saved during fine-tuning
model.load_weights('fine_tuned_best_model.h5')
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc:.4f} | Final Test Loss: {test_loss:.4f}")

# Plotting consolidated training history
def plot_history(history, history_fine):
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.axvline(x=INITIAL_EPOCHS-1, color='grey', linestyle='--', label='Start Fine-Tuning')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.axvline(x=INITIAL_EPOCHS-1, color='grey', linestyle='--', label='Start Fine-Tuning')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig('training_history_finetuned.png')
    plt.show()

plot_history(history, history_fine)
model.save('final_model_finetuned')
print("Training completed successfully!")