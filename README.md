![image description](/Users/noor/dropbox/github/Optimizer/logo.pdf)


# Variable Length Architecture Optimization

This project aims to enhance the architecture of a Convolutional Neural Network (CNN) utilized for genomics tasks by implementing a genetic algorithm (GA). Designing the architecture of a CNN can be a complex process due to the multiple parameters that must be configured, including activation functions, layer types, and hyperparameters. With the large number of parameters in many modern networks, finding the optimal configuration by hand is often impractical. Traditional methods can have limitations, and with limited computational resources available to researchers, many CNN projects are carried out by experts who rely on their specialized knowledge and experimentation results.

The GA approach to improving the architecture of CNNs removes the need for extensive expert knowledge and trial-and-error. This innovative method streamlines the process of optimizing CNNs and allows researchers to design optimal architectures with ease, regardless of their level of domain knowledge.

## 1. AO structure

- `main.py` : main function, use it to set the hyperparameters (i.e., learning rate, number of epochs). It also contains the main structure of the genetic algorithm

- `network.py` : contains the network class (i.e., invidual belonging to the population) and the transformations applied by the genetic algorithm (e.g., mutation)

- `topology.py` : contains the layer class (e.g., convolutional, pooling, dense). It also contains the block class, where each block refers to a group of sequential layers in a network

- `inout.py` : contains the input convolutional neural network that need to be optimized

- `utilities.py` : contains plot functions and common functions among the different files (e.g., load dataset)

#2. HO structure


### To run

To run the genetic algorithm:

```python3 HO/main.py```
