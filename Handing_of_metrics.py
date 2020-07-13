from tensorflow import keras
import tensorflow as tf 
from tensorflow.keras import layers
import time
# Get model
inputs = keras.Input(shape=(784,),name="digits")
x = layers.Dense(64,activation="relu",name="dense_1")(inputs)
x = layers.Dense(64,activation="relu",name="dense_2")(x)
outputs = layers.Dense(10,name="predictions")(x)
model = keras.Model(inputs=inputs,outputs=outputs)

optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

epochs = 2 
for epoch in range(epochs):
    print("\nStart of epoch %d" %(epoch,))
    start_time = time.time()

    for step, (x_batch_train,y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train,training=True)
            loss_value = loss_fn(y_batch_train,logits)
        grads = tape.gradient(loss_value,model.trainalbe_weights)
        optimizer.apply_gradients(zip(grads,model.trainable_weights))

        train_acc_metric.update_state(y_batch_train,logits)

        if step % 200 == 0:
            print("Training loss (for on batch) at step %d: %.4f"%(step,float(loss_value)))
            print("Seen so far: %d samples" %((step + 1 )*64))
    train_acc_metric.update_state(y_batch_train,logits)
    if step % 200 == 0:
        print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
        print("Seen so far: %d samples" % ((step + 1) * 64))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))