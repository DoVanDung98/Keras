import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import Sequential,layers

inputs = tf.keras.Input(shape=(3,3))
x = tf.keras.layers.Dense(4,activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5,activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs,outputs=outputs)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.dense1 = tf.keras.layers.Dense(4,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5,activation=tf.nn.softmax)
    def call(self,inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyModel()
