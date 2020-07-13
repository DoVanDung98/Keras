import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers


class CustomModel(keras.Model):
    def __init__(self,hidden_units):
        super(CustomModel,self).__init__()
        self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]
    def call(self,inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

def get_model():
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs,outputs)
    model.compile(optimizer="adam",loss="mean_squared_error")
    return model
    
model = CustomModel([16,16,10])
input_arr = tf.random.uniform((1,5))
outputs = model(input_arr)
model.save("my_model")

# Delete the custom-define model class to ensure that the loader does not have access to it
del CustomModel
loaded = keras.models.load_model("my_model")
np.testing.assert_allclose(loaded(input_arr),outputs)

print("Original model: ",model)
print("Load model: ",loaded)

model = get_model()
#Train the model
test_input = np.random.random((128,32))
test_target = np.random.random((128,1))

model.save("my_h5_model.h5")
reconstructed_model = keras.models.load_model("my_h5_model.h5")

# Let's check
np.testing.assert_allclose(
    model.predict(test_input),reconstructed_model.predict(test_input)
)

reconstructed_model.fit(test_input,test_target)