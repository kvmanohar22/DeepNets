import numpy as np
import matplotlib.pyplot as plt

# random number generator
rng = np.random

# generate random data
D = rng.randn(1000, 500)
# there are 10 hidden layers each having 500 units 
hidden_layer_sizes = [500] * 10
# all the layers have tanh non-linearities
nonlinearities_1 = ['tanh'] * len(hidden_layer_sizes)
nonlinearities_2 = ['relu'] * len(hidden_layer_sizes)
nonlinearities_3 = ['sigmoid'] * len(hidden_layer_sizes)


act = {'relu':lambda x:np.maximum(0, x), 'tanh': lambda x:np.tanh(x), 'sigmoid': lambda x:1./(1. + np.exp(-x))}
Hs = {}

# do the forward pass on all the 10 layers
for i in xrange(len(hidden_layer_sizes)):
	# input at this layer
	X = D if i==0 else Hs[i-1] 
	layer_in = X.shape[1]
	layer_out = hidden_layer_sizes[i]
	# layer initialization
	W =  rng.randn(layer_in, layer_out) / np.sqrt(layer_in)

	# matrix multiply
	H = np.dot(X, W)
	# induce the non-linearity
	H = act[nonlinearities_3[i]](H)	
	# cache the result on this layer
	Hs[i] = H

# look at the distributions at each layer
print 'Input layer had mean %f and std %f ' % (np.mean(D), np.std(D))
layer_means = [np.mean(H) for i, H in Hs.iteritems()]
layer_stds = [np.std(H) for i, H in Hs.iteritems()]
for i, H in Hs.iteritems():
	print 'Hidden layer %d had mean %f and std %f' % (i+1, layer_means[i], layer_stds[i])

# plot the means and std deviations
plt.figure()
plt.subplot(121)
plt.plot(Hs.keys(), layer_means, 'go-')
plt.title('layer mean')
plt.subplot(122)
plt.plot(Hs.keys(), layer_stds, 'bo-')
plt.title('layer std')

# plot the raw distributions
plt.figure()
for i, H in Hs.iteritems():
	plt.subplot(1, len(Hs), i+1)
	plt.hist(H.ravel(), 30, range=(-1, 1))

plt.show()	

