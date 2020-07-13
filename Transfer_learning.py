import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    keras.Input(shape=(784)),
    layers.Dense(32,activation="relu"),
    layers.Dense(32,activation="relu"),
    layers.Dense(32,activation="relu"),
    layers.Dense(10), 
])

model.load_weights(...)

# Freeze all layers except the last one
for layer in model.layers[:-1]:
    layer.trainable = False

# Recompile and train (this will only update the weights of the last layer)
model.compile(...)
model.fit(...)

# Load a convolutional base with pre-trained weights
base_model = keras.applications.Xception(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# Freeze the base model
base_model.trainable = False

# Use a Sequential model to add trainable calssifier on top
model = keras.Sequential([
    base_model,
    layers.Dense(1000),
])