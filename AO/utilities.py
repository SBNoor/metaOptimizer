from keras.utils import to_categorical
import pickle
import sys
from matplotlib import pyplot as plt
import numpy as np


def load_basset(batch_size, epochs, x_train_path , y_train_path, x_valid_path, y_valid_path):
    # we use memmap for training data to reduce the memory usage
    print('Loading Data')
    x_train, y_train = np.load(x_train_path), np.load(y_train_path)
    x_valid, y_valid = np.load(x_valid_path), np.load(y_valid_path)
    print(x_train.shape)
    x_train = np.squeeze(x_train)
    x_valid = np.squeeze(x_valid)
    print(x_train.shape)
    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    print(x_train.shape)
    dataset = {
        'batch_size': batch_size, ## 500
        'num_classes': len(y_train[0]), ## 164
        'epochs': epochs, ##20
        'x_train': x_train,
        'x_valid': x_valid,
        'y_train': y_train,
        'y_valid': y_valid
    }
    return dataset
    
    
def load_dataset(batch_size, epochs, x_train_path , y_train_path, x_valid_path, y_valid_path):  

    """
    dataset = {
        'batch_size': batch_size,
        'num_classes': num_classes,
        'epochs': epochs,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }
    """
    
    #if flag == 'basset':
    return load_basset(batch_size, epochs, x_train_path , y_train_path, x_valid_path, y_valid_path)
    
    #return dataset


def save_network(network):
    object_file = open('all_models/'+network.name + '.obj', 'wb')
    pickle.dump(network, object_file)


def load_network(name):
    object_file = open('all_models/'+name + '.obj', 'rb')
    return pickle.load(object_file)


def order_indexes(self):
    i = 0
    for block in self.block_list:
        block.index = i
        i += 1


def plot_training(history):                                           # plot diagnostic learning curves
    plt.figure(figsize=[8, 6])											# loss curves
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    filename = sys.argv[0].split('/')[-1]
    plt.savefig('graphs/'+filename + '_loss_plot.png')

    plt.figure(figsize=[8, 6])											# accuracy curves
    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)

    filename = sys.argv[0].split('/')[-1]
    plt.savefig('graphs/'+filename + '_acc_plot.png')
    plt.close()


def plot_statistics(stats):
    plt.figure(figsize=[8, 6])											# fitness curves
    plt.plot([s[0] for s in stats], 'r', linewidth=3.0)
    plt.plot([stats[0][0]] * len(stats), 'b', linewidth=3.0)
    plt.legend(['BestFitness', 'InitialFitness'], fontsize=18)
    plt.xlabel('Generations', fontsize=16)
    plt.ylabel('FitnessValue', fontsize=16)
    plt.title('Fitness Curve', fontsize=16)
    filename = sys.argv[0].split('/')[-1]
    plt.savefig('graphs/'+filename + '_fitness_plot.png')

    plt.figure(figsize=[8, 6])											# parameters curves
    plt.plot([s[1] for s in stats], 'r', linewidth=3.0)
    plt.plot([stats[0][1]] * len(stats), 'b', linewidth=3.0)
    plt.legend(['BestParamsNum', 'InitialParamsNum'], fontsize=18)
    plt.xlabel('Generations', fontsize=16)
    plt.ylabel('ParamsNum', fontsize=16)
    plt.title('Parameters Curve', fontsize=16)
    filename = sys.argv[0].split('/')[-1]
    plt.savefig('graphs/'+filename + '_params_plot.png')
    plt.close()
