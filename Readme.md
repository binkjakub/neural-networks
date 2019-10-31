# Neural Networks
Repository contains implementation of basic neural networks architectures using numpy only.
This code was prepared for neural network classes at Wroclaw University of Science and Technology. 

At the current state, implementation supports:
1. Architectures
    - [Multilayer Perceptron](src/nn/networks/mlp.py)
    - [Convolutaional Neural Network (CNN)]
   
1. Layers
    - [Linear](src/nn/layers/linear.py)
    - [Conv2d]
    - [MaxPool2d]
    
1. Activations
    - [Sigmoid](src/nn/activations/hidden_activations.py)
    - [TanH](src/nn/activations/hidden_activations.py)
    - [ReLu](src/nn/activations/hidden_activations.py)
    - [Softmax](src/nn/activations/softmax.py)
    
    
1. Losses
    - [Mean Squared Error](src/nn/losses/mse.py)
    - [Cross Entropy](src/nn/losses/cross_entropy.py)
    - [Cross Entropy Logit Loss]
    
1. Optimizers:
    - [Stochastic Gradient Descent (SGD)]
    - [Momentum]
    - [Adagrad]
    - [Adadelta]
    - [RMSprop]
    - [ADAM]
    
1. Initializers:
    - [Random Normal](src/nn/layers/initializers.py)
    - [Xavier](src/nn/layers/initializers.py)
    - [He](src/nn/layers/initializers.py)

NOTE:
*It is possible to easly extend architectures by composing existing layers or implement new ones.* 
## Implementation details 
Implementation utilizes OOP and computational graph approach (with manual gradient flow) and was inspired
by article: 
[Nothing but NumPy: Understanding & Creating Neural Networks with Computational Graphs from Scratch](https://medium.com/towards-artificial-intelligence/nothing-but-numpy-understanding-creating-neural-networks-with-computational-graphs-from-scratch-6299901091b0).
Under the scope of this project several study experiments on 
[MNIST](http://yann.lecun.com/exdb/mnist/) were also implemented. 
### Project structure
```
├── src
│   ├── data_processing
│   │   ├── notebooks
│   │   └── scripts
│   ├── datasets
│   ├── experiments
│   │   └── scripts
│   ├── nn
│   │   ├── activations
│   │   ├── layers
│   │   ├── losses
│   │   └── networks
│   └── utils
└── tests
```
---
Author: Jakub Binkowski
