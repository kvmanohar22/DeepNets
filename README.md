From Andrej Karpathy's course cs231n:CNNs for Visual Recognition

# How weight initialization affects the forward and backprop of a deep Neural Network ?
All the plots were generated with one full forward pass across all the `10` layers of the network with the same activation function
<br>
### Architecture
There are `10` layers, each layer having `500` units.
### Activation Functions
Tanh, ReLU, Sigmoid were used.
### Data
Random data points of `1000` training examples are generated from a univariate "normal" (Gaussian) distribution of mean `0` and variance `1`.
Weights for each layer were generated from the same distribution as that of `data points` but later on varied to obtain different plots.
