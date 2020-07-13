import tensorflow as tf 
from tensorflow import keras

class Linear(keras.layers.Layer):
    def __init__(self,units=32,input_dim=32):
        super(Linear,self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim,units),dtype="float32"),
            trainable=True, 
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,),dtype="float32"), trainable=True
        )
    def call(self,inputs):
        return tf.matmul(inputs,self.w) + self.b 

x = tf.ones((2,2))
linear_layer = Linear(4,2)
y = linear_layer(x)
print(y)

assert linear_layer.weights == [linear_layer.w,linear_layer.b]

class Linear(keras.layers.Layer):
    def __init__(self,units=32,input_dim=32):
        super(Linear,self).__init__()
        self.w =  self.add_weight(
            shape=(input_dim,units),initializer="random_normal",trainable=True
        )
        self.b = self.add_weight(shape=(units,),initializer="zeros",trainable=True)
    def call(self,inputs):
        return tf.matmul(inputs,self.w) + self.b

x = tf.ones((2,2))
linear_layer = Linear(4,2)
y = linear_layer(x)
print(y)

class ComputeSum(keras.layers.Layer):
    def __init__(self,input_dim):
        super(ComputeSum,self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),trainable=False)

    def call(self,inputs):
        self.total.assign_add(tf.reduce_sum(inputs,axis=0))
        return self.total
    
