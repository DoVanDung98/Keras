import tensorflow as tf 
from tensorflow import keras
import numpy as np 
class CustomModel(keras.Model):
    def train_step(self,data):
        x,y = data
        with tf.GradientTape() as tape:
            y_pred = self(x,training=True)
            loss = self.compiled_loss(y,y_pred,regularization_losses = self.losses)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss,trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y,y_pred)
        return {m.name:m.result() for m in self.metrics}


inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs,outputs)
model.compile(optimizer="adam",loss="mse",metrics=["mae"])

# just use `fit` as usal
x = np.random.random((1000,32))
y = np.random.random((1000,1))
# model.fit(x,y,epochs=3)

mae_metric = keras.metrics.MeanSquaredError(name="mae")
loss_tracker = keras.metrics.Mean(name="loss")

class CustomModel(keras.Model):
    def train_step(self,data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x,training=True)
            loss = keras.losses.mean_squared_error(y,y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss,trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_tracker.update_state(loss)
        mae_metric.update_state(y,y_pred)
        return {"loss": loss_tracker.result(),"mae":mae_metric.result()}

# construct an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs,outputs)

model.compile(optimizer="adam")

x = np.random.random((1000,32))
y = np.random.random((1000,1))
model.fit(x,y,epochs=100)
