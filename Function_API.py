import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,))

img_inputs = keras.Input(shape=(32,32,3))
print(inputs.shape)
print(inputs.dtype)

dense = layers.Dense(64,activation="relu")
x = dense(inputs)

x = layers.Dense(64,activation="relu")(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()

keras.utils.plot_model(model,"my_first_model.png")
keras.utils.plot_model(model,"my_first_model_with_shape_info.png",show_shapes=True)

(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32")/255
x_test = x_test.reshape(10000, 784).astype("float32")/255

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

test_scores = model.evaluate(x_test,y_test,verbose=2)
print("Test loss: ",test_scores[0])
print("Test accuracy: ",test_scores[1])

# model.save("model")
# del model 
# # Recreate the exact same model purely from file
# model = keras.models.load_model("model")

encoder_input = keras.Input(shape=(28,28,1),name="img")
x = layers.Conv2D(16,3,activation="relu")(encoder_input)
x = layers.Conv2D(32,3,activation="relu")(x)
x = layers.MaxPooling2D(3)(x) 
x = layers.Conv2D(32,3, activation="relu")(x)
x = layers.Conv2D(16,3,activation="relu")(x)
encoder_output=layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input,encoder_output,name="encoder")
encoder.summary()

x = layers.Reshape((4,4,1))(encoder_output)
x = layers.Conv2DTranspose(16,3,activation="relu")(x)
x = layers.Conv2DTranspose(32,3,activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16,3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1,3,activation="relu")(x)

autoencoder = keras.Model(encoder_input,decoder_output,name="autoencoder")
autoencoder.summary()

encoder_input = keras.Input(shape=(28,28,1), name="original_img")
x = layers.Conv2D(16,3,activation="relu")(encoder_input)
x = layers.Conv2D(32,3,activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32,3,activation="relu")(x)
x = layers.Conv2D(16,3,activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)
encoder.summary()

decoder_input = keras.Input(shape=(16,),name="encoded_img")
x = layers.Reshape((4,4,1))(decoder_input)
x = layers.Conv2DTranspose(32,3,activation="relu")(x)
x = layers.Conv2DTranspose(32,3,activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16,3,activation="relu")(x)

decoder_output = layers.Conv2DTranspose(1,3,activation="relu")(x)

decoder = keras.Model(decoder_input,decoder_output,name="decoder")
encoder.summary()

autoencoder_input = keras.Input(shape=(28,28,1),name="img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input,decoded_img,name="autoencoder")
autoencoder.summary()

