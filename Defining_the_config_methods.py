import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers

class CustomLayer(keras.layers.Layer):
    def __init__(self,a):
        self.var = tf.Variable(a,name="var_a")
    
    def call(self,inputs,training = False):
        if training:
            return inputs * self.var
        else:
            return inputs

    def get_config(self):
        return {"a":self.var.numpy()}

    @classmethod
    def from_config(cls,config):
        return cls(**config)

layer = CustomLayer(5)
layer.var.assign(2)

serialized_layer = keras.layers.serialize(layer)
new_layer = keras.layers.deserialize(
    serialized_layer,custom_objects={"CustomLayer":CustomLayer}
)

# Custom layer and function
class CustomLayers(keras.layers.Layer):
    def __init__(self,units=32,**kwargs):
        super(CustomLayers,self).__init__(**kwargs)
        self.units = units
    
    def build(self,input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape = (self.units,),initializer="random_normal",trainable=True
        )

    def call(self,inputs):
        return tf.matmul(inputs,self.w) + self.b
    
def custom_activation(x):
    return tf.nn.tanh(x) **2

# Make a model with the CustomLayer and custom_activation
inputs = keras.Input((32,))
x = CustomLayer(32)(inputs)

outputs = keras.layers.Activation(custom_activation)(x)
model = keras.Model(inputs,outputs)

config = model.get_config()

custom_objects = {"CustomLayer":CustomLayer,"custom_activation":custom_activation}
with keras.utils.custom_object_scope(custom_objects):
    new_model = keras.Model.from_config(config)

with keras.utils.custom_object_scope(custom_objects):
    new_model = keras.models.clone_model(model)

# Tranfering weights from one layer
def create_layer():
    layer = keras.layers.Dense(64,activation="relu",name="dense_2")
    layer.build((None,784))
    return layer

layer_1 = create_layer()
layer_2 = create_layer()

inputs = keras.Input(shape=(784,))