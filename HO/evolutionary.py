from copy import deepcopy
import random
import tensorflow as tf
import numpy as np

class Mutator:
    """
    Class for performing the mutation procedure

    :param nn_type: the type of NN architecture that you want to represent in
    encoded form (for example: 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    :param task: solving task ('regression' or 'classification')
    :param model: current NN model
    :param criterion: loss of current NN model
    :param optimizer: optimizer of current NN model
    :param batch_size: current batch size
    """

    def __init__(self, nn_type: str, task: str, model, criterion,
                 optimizer, batch_size):
        self.nn_type = nn_type
        self.task = task

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size

    def change_batch_size(self):
        """
        The method changes the size of the batch by a certain value

        :return model_dict: dictionary with model
        :return description: descriptions of changes
        """

        # Add a new number to current value
        new_batch_size = self.batch_size + random.randint(-10, 10)

        # Checking the validity of the solution
        if new_batch_size > 0:
            if new_batch_size < 200:
                pass
            else:
                new_batch_size = 200
        else:
            new_batch_size = self.batch_size + 2

        if new_batch_size != self.batch_size:
            description = ''.join(('batch size was changed from ',
                                   str(self.batch_size),' to ', str(new_batch_size)))
        else:
            description = 'nothing has changed'

        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': new_batch_size}

        return model_dict, description

    def change_layer_activations(self):
        """
        The method allows you to replace the activation function in the selected
         neural network layer

        """

        # Get layer names
        layer_names = [layer.name for layer in self.model.layers]

        amount_layers = len(layer_names)

        # Randomly choose layer by index
        layer_index = random.randint(0, amount_layers-1)
        random_layer = layer_names[layer_index]

        # Get available activations functions
        if layer_index == (amount_layers-1):
            act_functions = self._get_available_activations(is_this_layer_last=True)
        else:
            act_functions = self._get_available_activations(is_this_layer_last=False)

        # Randomly choose activation function (get name of it)
        new_name_function = random.choice(act_functions)

        # Get activation function object
        new_act_function = self._get_act_by_name(new_name_function)

        # Make changes
        self.model, prev_act = self._convert_activations(model=self.model,
                                                         layer_index=layer_index,
                                                         new_act_function=new_act_function)

        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': self.batch_size}

        if prev_act == new_name_function:
            description = 'nothing has changed'
        else:
            description = ''.join(('in layer with name ', str(random_layer),
                                   ' was changed activation function from ',
                                   prev_act, ' to ', new_name_function))

        return model_dict, description

    def change_optimizer(self):
        """
        The method changes the optimizer of NN model

        :return model_dict: dictionary with model
        :return description: descriptions of changes
        """

        # Find out the name of the optimizer
        current_opt = self.optimizer.__name__

        # Randomly select the optimizer
        available_optimizers = ['SGD', 'AdamW', 'Adam', 'Adadelta']
        random_optimizer = random.choice(available_optimizers)

        if random_optimizer == 'SGD' and random_optimizer != current_opt:
            description = ''.join(('optimizer was changed from ', current_opt, ' to SGD'))
            self.optimizer = tf.keras.optimizers.SGD
        elif random_optimizer == 'AdamW' and random_optimizer != current_opt:
            description = ''.join(('optimizer was changed from ', current_opt, ' to AdamW'))
            self.optimizer = tf.keras.optimizers.AdamW
        elif random_optimizer == 'Adam' and random_optimizer != current_opt:
            description = ''.join(('optimizer was changed from ', current_opt, ' to Adam'))
            self.optimizer = tf.keras.optimizers.Adam
        elif random_optimizer == 'Adadelta' and random_optimizer != current_opt:
            description = ''.join(('optimizer was changed from ', current_opt, ' to Adadelta'))
            self.optimizer = tf.keras.optimizers.Adadelta
        else:
            description = 'nothing has changed'
        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': self.batch_size}

        return model_dict, description

    def no_change(self):
        """
        Function dont do anything with NN model

        """
        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': self.batch_size}
        description = 'nothing has changed'

        return model_dict, description

    def change_neurons_activations(self):
        """
        TODO implement

        """
        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': self.batch_size}
        description = 'change layer neuron functions'

        return model_dict, description

    def change_loss_criterion(self):
        """
        TODO implement

        """

        model_dict = {'model': self.model, 'loss': self.criterion,
                      'optimizer': self.optimizer, 'batch': self.batch_size}
        description = 'change loss criterion'

        return model_dict, description

    def _get_available_activations(self, is_this_layer_last: bool) -> list:
        """
        The method returns the available activation functions for the selected task

        :param is_this_layer_last: is the layer for which activation functions
        are selected the final layer

        :return activations_list: list with names of an appropriate activation
        functions
        """

        if is_this_layer_last == True:
            if self.task == 'regression':
                activations_list = ['ELU',  'Hardsigmoid',
                                    'ReLU', 'SELU', 'Sigmoid', 'Tanh']
            elif self.task == 'classification':
                activations_list = ['Softmin', 'Softmax', 'LogSoftmax']
        else:
            activations_list = ['ELU',  'Hardsigmoid',
                                'ReLU', 'SELU', 'Sigmoid', 'Tanh']
        return activations_list

    def _convert_activations(self, model, layer_index, new_act_function):
        """
        The method replaces the activation function in the selected layer

        :param model: NN model to process
        :param layer_index: index of the layer where you want to replace the
        activation function
        :param name_function: activation function to be replaced

        :return model: the model is replaced by the activation function
        :return prev_act: name of the previous activation function in the layer
        TODO there is a need to make the function more adaptive
        """
        prev_act = str(model.layers[layer_index].activation)
        model.layers[layer_index].activation = new_act_function

        return model, prev_act

    def _get_act_by_name(self, name_function):
        """
        The method returns the corresponding function by it's name

        :param name_function: name of function
        :return fucntion_obj: new activation function
        """

        activations_dict = {
            'ELU': tf.keras.activations.elu,
            # 'Hardshrink': tf.keras.activations.hardshrink,
            'Hardsigmoid': tf.keras.activations.hard_sigmoid,
            # 'Hardtanh': tf.keras.activations.hard_tanh,
            'ReLU': tf.keras.activations.relu,
            # 'ReLU6': tf.keras.activations.relu6,
            'SELU': tf.keras.activations.selu,
            'Sigmoid': tf.keras.activations.sigmoid,
            'Tanh': tf.keras.activations.tanh,
            'Softmin': tf.keras.activations.softmax,  # Softmin doesn't have a direct equivalent, so using softmax
            'Softmax': tf.keras.activations.softmax,
            'LogSoftmax': tf.keras.activations.softmax,
        }

        fucntion_obj = activations_dict.get(name_function)
        return fucntion_obj


def generate_population(nn_type: str, task: str, actual_opt_path: str, actual_optimizer,
                        actual_criterion, actual_batch_size, actual_nn, amount_of_individuals: int,
                        check_mode: bool):
    """
    Method for generating a population

    :param nn_type: the type of NN architecture (for example: 'FNN', 'CNN', 'RNN', 'LSTM', 'AE')
    :param task: solving task ('regression' or 'classification')
    :param actual_opt_path: path to pth optimizer of current NN model
    :param actual_optimizer: current optimizer
    :param actual_criterion: loss of current NN model
    :param actual_batch_size: current batch size
    :param actual_nn: current NN model
    :param amount_of_individuals: number of individuals required
    :param check_mode: if True, in populations there is always one model
    will remain unchanged

    :return nns_list: list with neural network models as dict, where
        - model: neural network model
        - loss: loss function
        - optimizer: obtained optimizer
        - batch: batch size
    :return changes_list: list with descriptions of changes
    """

    nns_list = []
    changes_list = []
    for i in range(0, amount_of_individuals):
        # state = torch.load(actual_opt_path)

        # Make full copy of NN model
        actual_model = deepcopy(actual_nn)


        # Optimizer adaptive
        # optimizer_class = actual_optimizer.__class__
        # Different optimizers need different initialisation
        # if optimizer_class == tf.keras.optimizers.SGD:
        #     actual_optimizer = optimizer_class(actual_model.parameters(),
        #                                        lr=0.0001, momentum=0.9)
        # else:
        #     actual_optimizer = optimizer_class(actual_model.parameters())
        # actual_optimizer.load_state_dict(state['optimizer'])

        # Make copies for all parameters
        criterion_copy = deepcopy(actual_criterion)
        batch_copy = deepcopy(actual_batch_size)

        # Define mutation operator class
        mut_operator = Mutator(nn_type=nn_type,
                               task=task,
                               model=actual_model,
                               criterion=criterion_copy,
                               optimizer=actual_optimizer,
                               batch_size=batch_copy)

        # Make mutation
        # TODO implement 'change_loss_criterion', 'change_neurons_activations' operators
        operators = ['change_batch_size',
                     'change_layer_activations',
                     'change_optimizer']

        if check_mode == False:
            random_operator = random.choice(operators)
        else:
            # The model with index 0 in each population will be unchanged
            if i == 0:
                random_operator = 'no_change'
            else:
                random_operator = random.choice(operators)

        if random_operator == 'change_batch_size':
            mutated_model, change = mut_operator.change_batch_size()
        elif random_operator == 'change_loss_criterion':
            mutated_model, change = mut_operator.change_loss_criterion()
        elif random_operator == 'change_layer_activations':
            mutated_model, change = mut_operator.change_layer_activations()
        elif random_operator == 'change_neurons_activations':
            mutated_model, change = mut_operator.change_neurons_activations()
        elif random_operator == 'change_optimizer':
            mutated_model, change = mut_operator.change_optimizer()
        elif random_operator == 'no_change':
            mutated_model, change = mut_operator.no_change()

        nns_list.append(mutated_model)
        changes_list.append(change)

    return nns_list, changes_list


def eval_fitness(metadata: dict) -> list:
    """
    Function for evaluating the progress of multiple neural networks during training

    :param metadata: metadata about the population for a particular cycle in the
    form of a dictionary
        - key: model - index of the model (neural network)
        - value: list [a, b], where a - list with loss scores [...] and b -
        verbal description of what replacement was made in the neural network
        during mutation

    :return fitness_list: list with fitness scores
    """

    # Get metadata f
    models = list(metadata.keys())
    models.sort()

    # Calculate loss diff per epoch
    fitness_list = []
    for model in models:
        model_info = metadata.get(model)

        # Scores array with train losses
        scores_arr = model_info[0]

        # Calculate efficiency of NN
        start_loss = scores_arr[0]
        final_loss = scores_arr[-1]
        fitness = (start_loss - final_loss)-final_loss
        fitness_list.append(fitness)

    return fitness_list


def get_best_model(fitness_list: list, nns_list: list, crossover: bool):
    """
    The method allows you to get one model out of several by using either
    a selection operator or a crossover

    :param fitness_list: list with evaluated fitness scores for each model in nns_list
    :param nns_list: list with dict (NN models)
    :param crossover: is there a need to use crossover, if False, selection
    only will be used

    :return nn_model: dictionary with NN model
    """

    # If there is only 1 model in population
    if len(fitness_list) == 1:
        nn_model = nns_list[0]
    else:
        if crossover == True:
            # TODO Add ability to mix best models
            raise NotImplementedError("This functionality not implemented yet")
        else:
            # Selection: choose 1 the fittest model
            fitness_list = np.array(fitness_list)
            best_id = np.argmax(fitness_list)
            nn_model = nns_list[best_id]

            print(f'Best model after selection - index {best_id}')
    return nn_model
