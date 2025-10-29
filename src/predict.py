import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_SIZE, BATCH_SIZE, EPOCHS, NUM_CLASSES

def load_data(data_dir="data"):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    return train_ds, val_ds

def create_model():
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, (3, 3), activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu'), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    train_ds, val_ds = load_data()
    model = create_model()
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    model.save("models/pedigree_model.h5")
    print("✅ Model training complete. Saved as 'models/pedigree_model.h5'")
