import tensorflow as tf
import numpy as np
from model import NNOptimizer


X_train, Y_train = np.load("X_train_new.npy"), np.load("y_train_new.npy")
X_test, Y_test = np.load("X_valid_new.npy"), np.load("y_valid_new.npy")
print(X_train.shape)
X_train = np.squeeze(X_train)
X_test = np.squeeze(X_test)
print(X_train.shape)
X_train = X_train.transpose([0,2,1])
X_test = X_test.transpose([0,2,1])

# Initialization for classification task
# You need to prepare a class that will have "forward" function in addition to "initialization"

layers = [
    tf.keras.layers.Conv1D(filters=200,kernel_size=17,strides=1,padding='same',activation='relu',kernel_initializer='he_uniform',input_shape=X_train.shape[1:]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=200, activation='relu'),
    tf.keras.layers.Dense(units=len(Y_train[0]), activation='softmax')
]

FnnClassifier = tf.keras.Sequential(layers)


# Passing the prepared architecture to the optimizer class
gapfilling_optimizer = NNOptimizer(nn_type = 'FNN',
                                   task='classification',
                                   input=X_train,
                                   output=Y_train,
                                   cycles=2,
                                   population_size=2,
                                   epoch_per_cycle=2,
                                   fixing_epochs=2,
                                   runup_epochs=2,
                                   save_logs=True,
                                   logs_folder='models/')

# Find the best model hyperparameters set for chosen topology
best_solution = gapfilling_optimizer.optimize(source_nn=FnnClassifier,
                                              source_loss=tf.keras.losses.BinaryCrossentropy,
                                              source_optimizer=tf.keras.optimizers.Adam,
                                              source_batch_size=32,
                                              crossover=False,
                                              check_mode=False)
