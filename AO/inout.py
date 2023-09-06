import tensorflow as tf
import os
from utilities import load_network
from topology import Block, Convolutional, Pooling, Dropout, FullyConnected
from network import Network


def compute_parent(dataset, recreate_models, filters=320, filter_size=17, units=512):
    if os.path.isfile('all_models/'+'parent_0.h5') and not(recreate_models):
        daddy = load_network('parent_0')
        model = tf.keras.models.load_model('all_models/'+'parent_0.h5')
        print("Loading parent_0")
        print("SUMMARY OF", daddy.name)
        print(model.summary())
        print("FITNESS:", daddy.fitness)
        return daddy

    daddy = Network(0)

    layerList1 = [
        Convolutional(filters=filters, filter_size=filter_size, stride_size=1, padding='same',
                      input_shape=dataset['x_train'].shape[1:]),
        Convolutional(filters=filters, filter_size=filter_size, stride_size=1, padding='valid',
                      input_shape=dataset['x_train'].shape[1:])
    ]
    layerList2 = [
        Pooling(pool_size=2, stride_size=2, padding='same')
    ]
    daddy.block_list.append(Block(0, 0, layerList1, layerList2))
    '''
    layerList1 = [
        Convolutional(filters=64, filter_size=(3, 3), stride_size=(1, 1), padding='same',
                      input_shape=dataset['x_train'].shape[1:]),
        Convolutional(filters=64, filter_size=(3, 3), stride_size=(1, 1), padding='valid',
                      input_shape=dataset['x_train'].shape[1:])
    ]
    layerList2 = [
        Pooling(pool_size=(2, 2), stride_size=(2, 2), padding='same'),
        Dropout(rate=0.25)
    ]
    daddy.block_list.append(Block(1, 1, layerList1, layerList2))
    
    layerList1 = [
        Convolutional(filters=128, filter_size=(3, 3), stride_size=(1, 1), padding='same',
                      input_shape=dataset['x_train'].shape[1:]),
        Convolutional(filters=128, filter_size=(3, 3), stride_size=(1, 1), padding='valid',
                      input_shape=dataset['x_train'].shape[1:])
    ]
    layerList2 = [
        Pooling(pool_size=(2, 2), stride_size=(2, 2), padding='same'),
        Dropout(rate=0.25)
    ]
    daddy.block_list.append(Block(1, 1, layerList1, layerList2))
    '''
    layerList1 = [
        FullyConnected(units=units, num_classes=dataset['num_classes'])
    ]
    layerList2 = []
    daddy.block_list.append(Block(2, 1, layerList1, layerList2))

    model = daddy.build_model()
    daddy.train_and_evaluate(model, dataset)
    return daddy
