import tensorflow as tf  
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np 

inputs = keras.Input(shape=(784,),name="digits")
x1 = layers.Dense(64,activation="relu")(inputs)
x2 = layers.Dense(64,activation="relu")(x1)
outputs = layers.Dense(10,name="predictions")(x2)
model = keras.Model(inputs= inputs,outputs=outputs)

optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
batch_size = 64
(x_train, y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train,(-1,784))
x_test = np.reshape(x_train,(-1,784))
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)


epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" %(epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train,training=True)
            loss_value = loss_fn(y_batch_train,logits)
        grads = tape.gradient(loss_value,model.trainable_weights)
        optimizer.apply_gradients(zip(grads,model.trainable_weights))

        if step % 200 == 0:
            print("Training loss (for one batch) at step %d: %4f" %(step,float(loss_value)))
            print("Seen so far: %s samples"%((step+1) *64))

