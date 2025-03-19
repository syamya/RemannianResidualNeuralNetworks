import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback

# Step 1: Define dataset paths
train_dir = "./dataset/train"
test_dir = "./dataset/test"

# Step 2: Apply Guided Box Filtering (Preprocessing)
def guided_filter(image):
    return cv2.ximgproc.guidedFilter(image, image, radius=5, eps=10**-6)

# Step 3: Image preprocessing
image_size = (150, 150)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=guided_filter
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Step 4: Load MobileNetV2 as Encoder
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Step 5: Build the Model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)

predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Step 6: Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Custom Callback to Print Loss Every 10 Epochs
class LossPrinter(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Train Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")

# Step 8: Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    callbacks=[early_stopping, LossPrinter()]
)

# Step 9: Save the model
model.save('cavity_detection_model.keras')
print("Model saved as 'cavity_detection_model.keras'")

# Step 10: Visualize training results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()