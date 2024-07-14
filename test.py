import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

# Define constants
input_shape = (640, 480, 3)  # Adjust based on your image size
num_classes = 24  # Adjust based on the actual number of classes in your dataset

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\Lenovo\Downloads\IndianSIgnLAnguage\Data\train',
    target_size=(640, 480),
    batch_size=32,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    r'C:\Users\Lenovo\Downloads\IndianSIgnLAnguage\Data\test',
    target_size=(640, 480),
    batch_size=32,
    class_mode='categorical'
)

# Create model
model = create_model(input_shape, num_classes)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(test_generator)  # Adjust based on the number of validation batches
)

# Save model
model.save('indian_sign_language_model.h5')
