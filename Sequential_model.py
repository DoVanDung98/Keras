import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

# Denfine sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2,activation="relu",name="layer1"),
        layers.Dense(3,activation="relu",name="layer2"),
        layers.Dense(4,name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3,3))
y = model(x)

# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3,3))
y = layer3(layer2(layer1(x)))


model = keras.Sequential(
    [
        layers.Dense(2,activation="relu"),
        layers.Dense(3,activation="relu"),
        layers.Dense(4),
    ]
)
print("=================================")
print(model.layers)

model = keras.Sequential()
model.add(layers.Dense(2,activation="relu"))
model.add(layers.Dense(3,activation="relu"))
model.add(layers.Dense(4))
model.pop()
print(len(model.layers))

model = keras.Sequential(name="my_sequential")
model.add(layers.Dense(2,activation="relu",name="layer1"))
model.add(layers.Dense(3,activation="relu",name="layer2"))
model.add(layers.Dense(4,name="layer3"))

layer = layers.Dense(3)
print(layer.weights)

# Call layer on a test input
x = tf.ones((1,4))
y = layer(x)
print(layer.weights)

model = keras.Sequential(
    [
        layers.Dense(2,activation="relu"),
        layers.Dense(3,activation="relu"),
        layers.Dense(4),
    ]
) # No weights at this stage

x = tf.ones((1,4))
y = model(x)
print("Number of weights after calling the model: ",len(model.weights))

model.summary()

model = keras.Sequential()
model.add(keras.Input(shape=(4,)))
model.add(layers.Dense(2,activation="relu"))
model.summary()

model = keras.Sequential()
model.add(keras.Input(shape=(250,250,3))) # 250x250 RGB images
model.add(layers.Conv2D(32,5,strides=2,activation="relu"))
model.add(layers.Conv2D(32,3,activation="relu"))
model.add(layers.MaxPooling2D(3))

model.summary()

model.add(layers.Conv2D(32,3,activation="relu"))
model.add(layers.Conv2D(32,3,activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32,3,activation="relu"))
model.add(layers.Conv2D(32,3,activation="relu"))
model.add(layers.MaxPooling2D(2))

model.summary()

model.add(layers.GlobalMaxPooling2D())
model.add(layers.Dense(10))

initial_model = keras.Sequential(
    [
        keras.Input(shape=(250,250,3)),
        layers.Conv2D(32,5,strides=2,activation="relu"),
        layers.Conv2D(32,3,activation="relu"),
        layers.Conv2D(32,3,activation="relu"),
    ]
)
features_extractor = keras.Model(
    inputs = initial_model.inputs,
    outputs = [layer.output for layer in initial_model.layers],
)
x = tf.ones((1,250,250,3))
features = features_extractor(x)

initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)
# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)