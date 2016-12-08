<p>From Andrej Karpathy's course cs231n:CNNs for Visual Recognition</p>

<h1>How weight initialization affects the forward and backprop of a deep Neural Network ?</h1>
<p>All the plots were generated with one full forward pass across all the <code>10</code> layers of the network with the same activation function</p>
<br>
<li><h2>Architecture</h2></li>
<p>There are <code>10</code> layers, each layer having <code>500</code> units.</p>
<br>
<li><h2>Activation Functions</h2></li>
<p>Tanh, ReLU, Sigmoid were used.</p>
<br>
<li><h2>Data</h2></li>
<p>Random data points of <code>1000</code> training examples are generated from a univariate "normal" (Gaussian) distribution of mean <code>0</code> and variance <code>1</code> .</p>
<p>Weights for each layer were generated from the same distribution as that of <code>data points</code> but later on varied to obtain different plots.</p>
<br>
