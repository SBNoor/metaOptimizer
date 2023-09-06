import tensorflow as tf
import keras
import os
from network import Network
from inout import compute_parent
from random import randint, sample
from utilities import save_network,load_dataset, order_indexes, plot_training, plot_statistics, load_network
from copy import deepcopy
import pickle

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)      # suppress messages from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def initialize_population(population_size, dataset, recreate_models, filters, filter_size, units):
    print("----->Initializing Population")
    daddy = compute_parent(dataset, recreate_models, filters, filter_size, units)                                 # load parent from input
    population = [daddy]
    for it in range(1, population_size):
        population.append(daddy.asexual_reproduction(it, dataset, recreate_models))

    # sort population on ascending order based on fitness
    return sorted(population, key=lambda cnn: cnn.fitness)


def selection(k, population, num_population):
    if k == 0:                                              # elitism selection
        print("----->Elitism selection")
        return population[0], population[1]
    elif k == 1:                                            # tournament selection
        print("----->Tournament selection")
        i = randint(0, num_population - 1)
        j = i
        while j < num_population - 1:
            j += 1
            if randint(1, 100) <= 50:
                return population[i], population[j]
        return population[i], population[0]
    else:                                                   # proportionate selection
        print("----->Proportionate selection")
        cum_sum = 0
        for i in range(num_population):
            cum_sum += population[i].fitness
        perc_range = []
        for i in range(num_population):
            count = 100 - int(100 * population[i].fitness / cum_sum)
            for j in range(count):
                perc_range.append(i)
        i, j = sample(range(1, len(perc_range)), 2)
        while i == j:
            i, j = sample(range(1, len(perc_range)), 2)
        return population[perc_range[i]], population[perc_range[j]]


def crossover(parent1, parent2, it):
    print("----->Crossover")
    child = Network(it)

    first, second = None, None
    if randint(0, 1):
        first = parent1
        second = parent2
    else:
        first = parent2
        second = parent1

    child.block_list = deepcopy(first.block_list[:randint(1, len(first.block_list) - 1)]) \
                       + deepcopy(second.block_list[randint(1, len(second.block_list) - 1):])

    order_indexes(child)                            # order the indexes of the blocks

    return child


def genetic_algorithm(num_population, num_generation, num_offspring, dataset, recreate_models, filters, filter_size, units):
    print("Genetic Algorithm")

    population = initialize_population(num_population, dataset, recreate_models, filters, filter_size, units)

    print("\n-------------------------------------")
    print("Initial Population:")
    for cnn in population:
        print(cnn.name, ': ', cnn.fitness)
    print("--------------------------------------\n")

    # for printing statistics about fitness and number of parameters of the best individual
    stats = [(population[0].fitness, population[0].model.count_params())]

    for gen in range(1, num_generation + 1):

        '''
            k is the selection parameter:
                k = 0 -> elitism selection
                k = 1 -> tournament selection
                k = 2 -> proportionate selection
        '''
        k = randint(0, 2)

        print("\n------------------------------------")
        print("Generation", gen)
        print("-------------------------------------")

        for c in range(num_offspring):

            print("\nCreating Child", c)

            parent1, parent2 = selection(k, population, num_population)                 # selection
            print("Selected", parent1.name, "and", parent2.name, "for reproduction")

            child = crossover(parent1, parent2, c + num_population)                     # crossover
            print("Child has been created")

            print("----->Soft Mutation")
            child.layer_mutation(dataset)                                               # mutation
            child.parameters_mutation()
            print("Child has been mutated")

            model = child.build_model()                                                 # evaluation

            while model == -1:
                child = crossover(parent1, parent2, c + num_population)
                child.block_mutation(dataset)
                child.layer_mutation(dataset)
                child.parameters_mutation()
                model = child.build_model()

            child.train_and_evaluate(model, dataset)

            if child.fitness < population[-1].fitness:                                  # evolve population
                print("----->Evolution: Child", child.name, "with fitness", child.fitness, "replaces parent ", end="")
                print(population[-1].name, "with fitness", population[-1].fitness)
                # name = population[-1].name
                population[-1] = deepcopy(child)
                # population[-1].name = name
                population = sorted(population, key=lambda net: net.fitness)
            else:
                print("----->Evolution: Child", child.name, "with fitness", child.fitness, "is discarded")

        stats.append((population[0].fitness, population[0].model.count_params()))

    print("\n\n-------------------------------------")
    print("Final Population")
    print("-------------------------------------\n")
    for cnn in population:
        print(cnn.name, ': ', cnn.fitness)

    print("\n-------------------------------------")
    print("Stats")
    for i in range(len(stats)):
        print("Best individual at generation", i + 1, "has fitness", stats[i][0], "and parameters", stats[i][1])
    print("-------------------------------------\n")

    # plot the fitness and the number of parameters of the best individual at each iteration
    plot_statistics(stats)

    return population[0], population[:10]


def main():
    recreate = input('Please enter 1 for creating new population and 0 for using previous individuals: ')
    if '1' in recreate:
        recreate_models = True
    else:
        recreate_models = False

    # batch size will remain same for evolving iteration and final training for selected network
    # the number of training examples in one forward/backward pass
    batch = input('Please enter batch size for network input: ')
    try:
        batch_size = int(batch)
    except:
        batch_size = 20

    # Number of epochs for each individual's training in population during evolution
    # number of forward and backward passes of all the training examples
    epc = input('Please enter number of epochs for evolving network training: ')
    try:
        epochs = int(epc)
    except:
        epochs = 20
    
    # Total size of population
    pop_size = input('Please enter population size: ')
    try:
        num_population = int(pop_size)
    except:
        num_population = 20
    
    # Total number of evolving generations
    gen = input('Please enter number of Total generations: ')
    try:
        num_generation = int(gen)
    except:
        num_generation = 20
    
    # Total number of offsprings in each generation to compete native population
    ofspri = input('Please enter number of offsprings in each generation to compete native population: ')
    try:
        num_offspring = int(ofspri)
    except:
        num_offspring = 20
    
    # filters, filter_size, units
    # Total number of filters in parent's convolution layers
    filters = input("Please enter number of filters in parent's convolution layers: ")
    try:
        filters = int(filters)
    except:
        filters = 320
    
    # Total filter size of convolution filter
    filter_size = input('Please enter size of convolution filter: ')
    try:
        filter_size = int(filter_size)
    except:
        filter_size = 17
    
    # Total number of neurons in fully connected layer for initial parents
    units = input('Please enter number of neurons in fully connected layer for parent in initial population: ')
    try:
        units = int(units)
    except:
        units = 512


    # Change the file name here for using different data sets
    x_train_path, y_train_path = "X_train_new.npy", "y_train_new.npy"
    x_valid_path, y_valid_path = "X_valid_new.npy", "y_valid_new.npy"
    dataset = load_dataset(batch_size, epochs, x_train_path , y_train_path, x_valid_path, y_valid_path)

    

    # plot the best model obtained
    optCNN, optCNN_top10 = genetic_algorithm(num_population, num_generation, num_offspring, dataset, recreate_models, filters, filter_size, units)

    for x,i in enumerate(optCNN_top10):
        i.model.save('top10ind/'+f'{x}_model.h5')
        object_file = open('top10ind/'+f'{x}_model' + '.obj', 'wb')
        pickle.dump(i, object_file)

    # plot the training and validation loss and accuracy
    final_training_epoch = 3
    model = optCNN.build_model()
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy',auroc])
    history = model.fit(dataset['x_train'],
                        dataset['y_train'],
                        batch_size=dataset['batch_size'],
                        epochs=final_training_epoch,
                        validation_data=(dataset['x_valid'], dataset['y_valid']),
                        shuffle=True)
    optCNN.model = model                                        # model
    optCNN.fitness = history.history['val_loss'][-1]            # fitness

    # Saving fully trained best fit individual
    optCNN.model.save('best_fit_trained/'+f'{optCNN.name}_model.h5')
    object_file_optCNN = open('best_fit_trained/'+optCNN.name + '.obj', 'wb')
    pickle.dump(i, object_file_optCNN)


    print("\n\n-------------------------------------")
    print("The initial CNN has been evolved successfully in the individual", optCNN.name)
    print("-------------------------------------\n")
    daddy = load_network('parent_0')
    model = tf.keras.models.load_model('all_models/'+'parent_0.h5')
    print("\n\n-------------------------------------")
    print("Summary of initial CNN")
    print(model.summary())
    print("Fitness of initial CNN:", daddy.fitness)

    print("\n\n-------------------------------------")
    print("Summary of evolved individual")
    print(optCNN.model.summary())
    print("Fitness of the evolved individual:", optCNN.fitness)
    print("-------------------------------------\n")

    plot_training(history)


if __name__ == '__main__':
    main()
