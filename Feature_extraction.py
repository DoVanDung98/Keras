import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers

initial_model = keras.Sequential(
    [
        keras.Input(shape=(250,250,3)),
        layers.Conv2D(32,5,strides=2,activation="relu"),
        layers.Conv2D(32,3,activation="relu"),
        layers.Conv2D(32,3,activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs = initial_model.inputs,
    outputs = [layer.output for layer in initial_model.layers],
)
x = tf.ones((1,250,250,3))
features = feature_extractor(x)

initial_model = keras.Sequential(
    [
        keras.Input(shape=(250,250,3)),
        layers.Conv2D(32,5,strides=2,activation="relu"),
        layers.Conv2D(32,3,activation="relu",name="my_intermediate_layer"),
        layers.Conv2D(32,3,activation="relu"),
    ]
)
features_extractor = keras.Model(
    inputs = initial_model.inputs,
    outputs = initial_model.get_layer(name="my_intermediate_layer").output,
)
# Call feature extractor on test input
x = tf.ones((1,250,250,3))
features = features_extractor(x)