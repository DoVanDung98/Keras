import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np 

image_input = keras.Input(shape=(32,32,3),name= "img_input")
timeseries_input = keras.Input(shape=(None,10),name="ts_input")

x1 = layers.Conv2D(3,3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)

x2 = layers.Conv1D(3,3)(timeseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)

x = layers.concatenate([x1,x2])

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

def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model
score_output = layers.Dense(1,name="score_output")(x)
class_output = layers.Dense(5, activation="softmax",name="class_output")(x)

model = keras.Model(
    inputs = [image_input,timeseries_input],
    outputs = [score_output,class_output]
)
keras.utils.plot_model(model,"model_images/multi_input_and_output_model.png",show_shapes=True)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss = [keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()],
    metrics = [
        [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError(),
        ],
        [keras.metrics.CategoricalAccuracy()],
    ]
)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "score_output": keras.losses.MeanSquaredError(),
        "class_output": keras.losses.CategoricalCrossentropy(),
    },
    metrics={
        "score_output": [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError(),
        ],
        "class_output": [keras.metrics.CategoricalAccuracy()],
    },
)

model.compile(
    optimizer = keras.optimizers.RMSprop(1e-3),
    loss=[None,keras.losses.CategoricalCrossentropy()],
)

model.compile(
    optimizer = keras.optimizers.RMSprop(1e-3),
    loss={"class_output":keras.losses.CategoricalCrossentropy()},
)

model.compile(
    optimizer = keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(),keras.losses.CategoricalCrossentropy()],
)

# Generate dummy NumPy data
img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))

# Fit on lists
model.fit([img_data, ts_data], [score_targets, class_targets], batch_size=32, epochs=1)

# Alternatively, fit on dicts
model.fit(
    {"img_input": img_data, "ts_input": ts_data},
    {"score_output": score_targets, "class_output": class_targets},
    batch_size=32,
    epochs=1,
)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"img_input": img_data,"ts_input":ts_data},
        {"score_output":score_targets,"class_output":class_targets},
    )
)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
model.fit(train_dataset,epochs=10)

model = get_compiled_model()

callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor= "val_loss",
        # "no longger improveing" being defined as "no better than 1e-2 less"
        min_delta = 1e-2,
        patience=2,
        verbose=1,
    )
]
model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=64,
    callbacks=callbacks,
    validation_split=0.2,
)

