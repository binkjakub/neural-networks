# Neural Networks
Repository contains implementation of basic neural networks architectures using numpy only.
This code was prepared for neural network classes at Wroclaw University of Science and Technology. 

At the current state, implementation supports:
1. Architectures
    - [Multilayer Perceptron](src/nn/networks/mlp.py)
    
2. Activations
    - [Sigmoid](src/nn/activations/sigmoid.py)
    - [Softmax](src/nn/activations/softmax.py)
    
3. Losses
    - [Mean Squared Error](src/nn/losses/mse.py)
    - [Cross Entropy](src/nn/losses/cross_entropy.py)
    
4. Optimizers:
    - Stochastic Gradient Descent
    
5. Initializers:
    - Random Normal

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
│   ├── learning
│   ├── nn
│   │   ├── activations
│   │   ├── layers
│   │   ├── losses
│   │   └── networks
│   └── utils
└── tests
```