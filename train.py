import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==== PARAMETER ====
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
RANDOM_SEED = 42

# ==== PATH DATA ====
BASE_DIR = os.path.join(
    "dataset",
    "Augmented IQ-OTHNCCD lung cancer dataset"
)

if not os.path.isdir(BASE_DIR):
    raise FileNotFoundError(f"Folder dataset tidak ditemukan: {BASE_DIR}")

print("Dataset base dir:", BASE_DIR)

# ==== LOAD DATASET ====
train_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR,
    validation_split=0.2,
    subset="training",
    seed=RANDOM_SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR,
    validation_split=0.2,
    subset="validation",
    seed=RANDOM_SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Kelas:", class_names)

# Prefetch untuk percepat training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ==== AUGMENTASI & PREPROCESS ====
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)

# ==== MODEL TRANSFER LEARNING ====
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)
x = preprocess_input(x)

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=x
)
base_model.trainable = False  # freeze dulu

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs, name="lung_cancer_classifier")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==== TRAINING ====
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "models/best_lung_cancer_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# ==== SIMPAN MODEL FINAL ====
os.makedirs("models", exist_ok=True)
model.save("models/lung_cancer_model.h5")
print("Model disimpan di models/lung_cancer_model.h5")
print("Best model (berdasar val_accuracy) di models/best_lung_cancer_model.h5")
