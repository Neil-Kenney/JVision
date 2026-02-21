import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "gesture_data"
SEQ_LENGTH = 20
FEATURE_SIZE = 63  # 21 landmarks × xyz

# -----------------------------
# LOAD + VALIDATE DATA
# -----------------------------
X = []
y = []

print("Loading dataset...")

for file in os.listdir(DATA_DIR):
    if not file.endswith(".npy"):
        continue

    # Correct label extraction: swipe_left_123.npy → swipe_left
    label = "_".join(file.split("_")[:-1])

    sequence = np.load(os.path.join(DATA_DIR, file))

    # Validate shape
    if sequence.shape != (SEQ_LENGTH, FEATURE_SIZE):
        print(f"Skipping malformed file: {file} shape={sequence.shape}")
        continue

    X.append(sequence)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("\nDataset loaded:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# ENCODE LABELS
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

print("\nClasses:", label_encoder.classes_)

# -----------------------------
# TRAIN/VAL SPLIT
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, shuffle=True
)

# -----------------------------
# BUILD LSTM MODEL
# -----------------------------
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, FEATURE_SIZE)),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# TRAIN MODEL
# -----------------------------
checkpoint = ModelCheckpoint(
    "gesture_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=[checkpoint]
)

print("\nTraining complete. Best model saved as gesture_model.h5")

# -----------------------------
# SAVE LABEL ENCODER
# -----------------------------
np.save("gesture_labels.npy", label_encoder.classes_)
print("Saved label classes to gesture_labels.npy")
