from tensorflow import keras
model = keras.models.load_model("model")
import numpy as np 
import tensorflow as tf 
from tensorflow import keras

def get_model():
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs,outputs)
    model.compile(optimizer="adam",loss="mean_squared_error")
    return model

model = get_model()
test_input = np.random.random((128,32))
test_target = np.random.random((128,1))

model.fit(test_input,test_target)
model.save("my_model")

# It can be used to reconstruct the model identically
reconstructed_model = keras.models.load_model("my_model")

np.testing.assert_allclose(
    model.predict(test_input),reconstructed_model.predict(test_input)
)

reconstructed_model.fit(test_input,test_target)