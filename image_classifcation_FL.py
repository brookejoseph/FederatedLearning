import tensorflow as tf 
import tensorflow_federated as tff
from tensorflow_federated import paillier
import numpy as np 
import collections
import functools
import os
import time
import random 
import flwr as fl 

#making sure we produce the same results everytime we run the same example 
np.random.seed(0)
from tensorflow.python.keras.optimizer_v2 import gradient_descent 
from tensorflow_federated import python as tff 
from random import choices 

#
NUM_EPOCHES = 5
BATCH_SIZE = 20 
SHUFFLE_BUFFER = 500 
NUM_CLIENTS = 3

#enabling the tensorflow version 2 behavior to make sure that this will all work 
tf.compact.v1.enable_v2_behavior()

#here we are just trying to simulate the types of datasets we would receive froma federated learning model 
#tff already has some data sets that we can use
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

emnist_test.head()
#here we are reshaping the data we recieved, and we are setting the pixels equal to x and labels equal to y 
#reshaping the tensor to -1 simply means we want it to be 1 dimensional there for this example rather than having a 28x28 dataset we have a 784 element array
#OrderedDict, simply is a way of remebereing the order in which the dat was collected 
def preprocess(dataset):
    def element_fn(element):
        return collections.OrderedDict([
        ('x', tf.reshape(element['pixels'], [-1])),
        ('y', tf.reshape(element['label'], [1])),
        ])
    return dataset.repeat(NUM_EPOCHES).map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)

#its important to note that a efficient way of feeding data to tff in a simulation is simply as a python list 
#each element of the list should hold the data of an indivdual user
# the data can also be entered with tf.data.Dataset 
def make_federated_data(client_data, client_ids):
    return[preprocess(client_data.create_tf_dataset_for_client(x))
    for x in client_ids]

sample_clients = emnist_train.client_ids[0: NUM_CLIENTS]
#going through ID we said and it is going to donate the tensorflow dataset object


#we are selcting 10 of our clients and we have prepocrssed the datsets in a way we picked 
federated_train_data = make_federated_data(emnist_train, sample_clients)
print(f'Number of clinet datasets:{len(federated_train_data)}')
print(f'First dataset:{federated_train_data[0]}')

sample_clients = emnist_test.client_ids[0: NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_test, sample_clients)

#here we are providing a batch of our traning dataset, then we reconsturct an x and y 
sample_batch = iter(federated_train_data[0]).next()
sample_batch = collections.OrderedDict([
    ('x', sample_batch['x'].numpy()),
    ('y', sample_batch['y'].numpy()),
])

#here we are just creating a Keras model needed
def create_keras_model():
    return tf.keras.models.Sequntial([
        tf.keras.layers.Input.Layer(input_shapre=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer= 'zeros'),
        tf.keras.layers.Softmax(),
    ])

#centrlized training 



#here we are creating a new model we pass in the model, then an input sepc to give tff an example of the data they're consuming 
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocess_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy
        metric=[tf.keras.metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.build_federated_averaging_process(
    client_optimizer_fn=lambda: tf.keras.optimizers.SDG(learning_rate=0.2),
)

server_optimizer_fn=lambda: tf.keras.optimizers.SDG(learning_rate=0.5)

state = iterative_process.initialize()
state,metrics = iterative_process.next(state, federated_train_data)
print('round `, metrics={}'.format(metrics['train']))

NUM_ROUNDS = 11 
for round_num in range(2,NUM_ROUNDS):
    state, metrics = iterative_process.next(state,federated_train_data)
    print('round{:2d}, metrics{}'.format(round_num,metrics['train']))

evaluation = tff.learning.build_federated_evaluation(model_fn)
shuffled_ids = emnist_test.client_ids.copy()
random.shuffle(shuffled_ids)
sample_clients - shuffled_ids[0:NUM_CLIENTS]

federated_test_data = make_federated_data(emnist_test, sample_clients)

len(federated_test_data), federated_test_data[0]

test_metrics = evaluation(state.model, federated_test_data)
