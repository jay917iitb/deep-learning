import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = r"C:\Users\yoga\Documents\deep_learning\tensor flow tut\CUB_200_2011"
IMAGE_DIR = os.path.join(BASE_DIR, 'images')

# Load dataset with improved validation split
def load_data(val_split=0.2):
    images_txt = pd.read_csv(os.path.join(BASE_DIR, 'images.txt'), sep=' ', header=None, names=['img_id', 'filepath'])
    labels_txt = pd.read_csv(os.path.join(BASE_DIR, 'image_class_labels.txt'), sep=' ', header=None, names=['img_id', 'label'])
    split_txt = pd.read_csv(os.path.join(BASE_DIR, 'train_test_split.txt'), sep=' ', header=None, names=['img_id', 'is_train'])

    data = images_txt.merge(labels_txt, on='img_id').merge(split_txt, on='img_id')
    data['filepath'] = data['filepath'].apply(lambda x: os.path.join(IMAGE_DIR, x))
    data['label'] -= 1  # class labels: 0 to 199
    
    # Split into train, validation and test
    train_val_df = data[data['is_train'] == 1]
    test_df = data[data['is_train'] == 0]
    
    # Further split train into train and validation
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_split, 
        stratify=train_val_df['label'],
        random_state=42
    )
    
    return train_df, val_df, test_df

train_df, val_df, test_df = load_data()

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 200
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 100
INIT_LR = 3e-4

# Enhanced data augmentation
def augment_image(image):
    # Random flips
    image = tf.image.random_flip_left_right(image)
    
    # Random rotation
    if tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image)
    
    # Random brightness, contrast, saturation
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    
    # Random zoom and crop
    if tf.random.uniform(()) > 0.5:
        scale = tf.random.uniform([], 1.0, 1.2)  # Only zoom out
        new_h = tf.cast(scale * IMG_SIZE[0], tf.int32)
        new_w = tf.cast(scale * IMG_SIZE[1], tf.int32)
        image = tf.image.resize(image, (new_h, new_w))
        image = tf.image.random_crop(image, size=[IMG_SIZE[0], IMG_SIZE[1], 3])
    
    # Random quality
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_jpeg_quality(image, 75, 100)
    
    return image

def preprocess(path, label, training=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    
    if training:
        image = augment_image(image)
    
    # Normalize with ImageNet stats
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

def prepare_dataset(df, training=False):
    paths = df['filepath'].values
    labels = df['label'].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda x, y: preprocess(x, y, training), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(2000, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = prepare_dataset(train_df, training=True)
val_ds = prepare_dataset(val_df, training=False)
test_ds = prepare_dataset(test_df, training=False)

# Create a smaller ResNet-like model with ~10M parameters
def create_light_resnet():
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    
    # Initial stem
    x = layers.Conv2D(32, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks with reduced filters
    def residual_block(x, filters, stride=1):
        shortcut = x
        
        x = layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x
    
    # Stack of residual blocks
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    # Final layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

model = create_light_resnet()
model.summary()

# Enhanced training configuration
def lr_schedule(epoch):
    """Learning rate scheduler with warmup"""
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return INIT_LR * (epoch + 1) / warmup_epochs
    return INIT_LR * 0.5 * (1 + tf.math.cos(epoch * 3.1416 / EPOCHS))

# Improved callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    ),
    keras.callbacks.EarlyStopping(
        patience=15,
        restore_best_weights=True,
        monitor='val_accuracy',
        mode='max'
    ),
    keras.callbacks.LearningRateScheduler(lr_schedule),
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]

# Compile with label smoothing
def smoothed_categorical_crossentropy(y_true, y_pred, smoothing=0.1):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=NUM_CLASSES)
    y_true = y_true * (1.0 - smoothing) + smoothing / NUM_CLASSES
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=INIT_LR, weight_decay=1e-4),
    loss=smoothed_categorical_crossentropy,
    metrics=['accuracy']
)

# Training with validation
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

# Final evaluation on test set
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Save final model
model.save('final_model')
print("Training completed successfully!")