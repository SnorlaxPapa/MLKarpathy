# Problem

By default, when we initialize weights and biases, the distribution of the value of the weights is skewed unless we use a distribution of elements. This means that initial answers could be heavily skewed, leading to massive activation deadzones in the activation layer (affecting gradient backflow). Softmax layer could also be confidently wrong, leading to wasted training steps to correct this initial skew in distribution.

Additionally, if your variance is too small in your weights, activations will barely vary across different inputs, leading to poor back-prop gradient-flow

--- 

Similarly, the variance i when wegrowsghts are multiplied and added together, leading to an even more uneven distribution. 

---

# Fix

In order to prevent this, we aim to standardize the variance in the weights, and one way to do this is by scaling the weights in the proper manner. 

---

## Xavier/Glorot
In order to do this, our objective should be mapping `Var(Y) == Var(X)`, where Y is the variance of the output and X is the variance of the input (i.e. the sum of wixi for i in input + b). The math for Xavier/Glorot's derivation is simply solving for variance of the weight such that it maintains the variance in i/o layers, which resolved to `1/n_in` in a forward pass. In considering the backward pass, when gradients flow backwards, there is also a variance in gradients as they are summed and multiplied, which resolves to 1/n_out. Xavier/Glorot's paper hence takes the compromise Var(w) = 2/ (n_in + n_out)

---

## Kai Ming
For ReLU, it is expected to zero out exactly half of all values in a Gaussian distribution with mean = 0 after the weights (as with many other activations as they 'squash' values), halving the variance. So on top of Xavier/Glorot scale, we need to implement an additional gain factor that counteracts the drop in number of elements. In the case of relu, gain is sqrt(2). He Kai Ming's paper

Default on Pytorch is based on Kai Ming's approach with fan_in (stability for forward passes). 

---

## Batch normalization

A common issue with traditional neural networks is internal covariant shifting. That is, each layer is fitting to a moving target as layers are updated simultaenously. The first pass might tell the 1st layer to fit to x for the second layer, but the second layer would have simultaneously fit to Z for the third layer and so on. If the second layer fit too much and it's distribution and values drastically change, then the fitting of layer 1 would have been for naught.
To slow down this shift, batch normalization fixes the scale of each neuron. 

It takes a mini-batch, and uses a biased estimator for the population meana and variance and using those values to standardize the neurons in this mini-batch. A bngain multiplicative scalar and bnbias additive  is added for each neuron in the form bn_gain * (output - output.mean ) / output.std + bn_bias. Where bn_gain and bn_bias adapt to each individual neuron's activations

For inference, we don't have mini-batches to use, so instead we track the rolling std and mean during training, and use that as our scalars during inference time when our batch size could potentially be v small.

A key thing to note is that due to the calculaation of output.mean and std, the output of the logits is coupled downstream with all the training examples in that mini-batch. Surprisingly, this acts as a regularization feature and noises the logits, improving performance. 

You don't need a bias in your layer itself because batch_norm has a bias by itself per neuron!

Or does BN actually fix internal covariant shift? Maybe, maybe not.