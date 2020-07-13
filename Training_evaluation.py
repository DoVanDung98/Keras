import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784),name="digits")
x = layers.Dense(64,activation="relu",name="dense_1")(inputs)
x = layers.Dense(64,activation="relu",name="dense_2")(x)
outputs = layers.Dense(10,activation="softmax",name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()

# Preprocessing the data (these are Numpy arrays)
x_train = x_train.reshape(60000,784).astype("float32")/255
x_test = x_test.reshape(10000,784).astype("float32")/255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics = [keras.metrics.SparseCategoricalAccuracy()],
)

print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=10,
    validation_data=(x_val,y_val),
)
print(history.history)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")

results = model.evaluate(x_test,y_test,batch_size=128)
print("test loss, test acc: ",results)

print("Generate predictions for 3 sample")
predictions = model.predict(x_test[:3])
print("prediction shape: ",predictions.shape)

model.compile(
    optimizer = keras.optimizers.RMSprop(learning_rate=1e-3),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = [keras.metrics.SparseCategoricalAccuracy()],
)
model.compile(
    optimizer="rmsprop",
    loss = "saperse_categorical_crossentropy",
    metrics= ["sparse_categorical_accuracy"],
)

def get_uncompiled_model():
    inputs = keras.Input(shape=(784,),name="digits")
    x = layers.Dense(64,activation="relu",name="dense_1")(inputs)
    x = layers.Dense(64,activation="relu",name="dense_2")(x)
    outputs = layers.Dense(10,activation="softmax",name="predictions")
    return model 

def get_compiled_model():
    model = get_compiled_model()
    model.compile(
        optimizer = "rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics = ["sparse_categorical_accuracy"],
    )
    return model 

def custom_mean_squared_error(y_true,y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(),loss=custom_mean_squared_error)

# We need to one-hot encode the labels to use MSE
y_train_one_hot = tf.one_hot(y_train,depth=10)
model.fit(x_train,y_train_one_hot, batch_size=64,epochs=10)

class CustomMSE(keras.losses.Loss):
    def __init__(self,regularization_factor=0.1,name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor
    
    def call(self, y_true,y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true-y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5-y_pred))
        return mse + reg * self.regularization_factor 
model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(),loss=CustomMSE())

y_train_one_hot = tf.one_hot(y_train,depth=10)
model.fit(x_train,y_train_one_hot, batch_size=64,epochs=1)


class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)


model = get_uncompiled_model()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[CategoricalTruePositives()],
)
model.fit(x_train, y_train, batch_size=64, epochs=3)