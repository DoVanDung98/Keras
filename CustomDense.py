import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

class CustomDense(layers.Layer):
    def __init__(self,units=32):
        super(CustomDense,self).__init__()
        self.units = units
    
    def build(self,input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )
    def call(self,inputs):
        return tf.matmul(inputs, self.w) + self.b

inputs = keras.Input((4,))
ouputs = CustomDense(10)(inputs)

model = keras.Model(inputs,ouputs)
