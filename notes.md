# 1. Neural Networks and Deep Learning
---

## Week 1: 

- Scale drives deep learning progress: Increase in labelled data, computational power and better algorithms have made deep learning take off. 

- The slope of the sigmoid function is close to zero for really small and really large input values. This makes relu a better choice, since **gradient descent becomes faster**.[Video reference](https://www.coursera.org/learn/neural-networks-deep-learning/lecture/praGm/why-is-deep-learning-taking-off)

## Week 2: 

- One reason that *mean squared error* is not used in the cost function of logistic regression is that doing so makes the cost function non-convex. [Video reference](https://www.coursera.org/learn/neural-networks-deep-learning/lecture/yWaRd/logistic-regression-cost-function)

- **Broadcasting**: `A(m, n) +,-,*,/ B(1, n)`
This reshapes B into (m, n) matrix and does element-wise operation. [Video reference](https://www.coursera.org/learn/neural-networks-deep-learning/lecture/uBuTv/broadcasting-in-python)

- **Rank-1 array v/s vector**: 
```
import numpy as np
a = np.random.randn(5)    # Rank-1 array
a.shape                   # (5,)

b = np.random.randn(5, 1) # Column vector
c = np.random.randn(1, 5) # Row vector

assert(c.shape == (1, 5)) # True
```
Defining explicitly (like `b` or `c`) avoids errors.


# Improving Deep Neural Networks
---

## Week 1:

#### Much of the stuff in this week is discussed in Yann Lecun's excellent paper [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf). Must read! 

|               |     Model    | Train error | Test error | How to resolve                                       |
|:-------------:|:------------:|:-----------:|:----------:|------------------------------------------------------|
|   High Bias   | Underfitting |     15%     |     16%    | Bigger network, Train longer, Change hyperparameters |
| High Variance |  Overfitting |      1%     |     11%    | More data, Regularization                            |
|      Both     |              |     15%     |     30%    |                                                      |
|    Neither    |              |      1%     |     2%     |                                                      |

This is assuming very low Bayes error, i.e., with best available or human score.Otherwise consider Train error - Bayes error.

Assumption: Both training set and test set are from the same distribution.

- Bias-Variance tradeoff: Used to exist in pre-Deep Learning days. But now, using bigger network and getting more data can help in reducing both - bias and variance. So no longer a trade-off. One reason why Deep Learning has taken off for supervised learning. [Video reference](https://www.coursera.org/learn/deep-neural-network/lecture/ZBkx4/basic-recipe-for-machine-learning)

- Why not regularize the bias term? In practice there are many input parameters and hidden nodes so not adding the regularization to _one_ of the terms (intercept `b`) shouldn't affect the overall training. [Video reference](https://www.coursera.org/learn/deep-neural-network/lecture/Srsrc/regularization) 

- L^1 v/s L^2 regularization: Using L^1 regularization makes the weight vector sparse. (Why?) [Same video as above]

- **L^2 regularization is Weight Decay**: The expression for updating the weight `w` is 
```
w^[l] := w^[l] - alpha*lambda/m w^[l] - alpha(from BackProp)
```
This reduction in `w^[l]` is referred to as *weight decay*.

- **Regularization prevents overfitting (Intuition ONLY)**: 
    1. If lambda > 0 then weights w tend to be small, which is equivalent to turning off many neurons, which makes the model closer to being linear. 
    2. If lambda > 0 then weights w tend to be small, so in the equation 
        `z = w*a + b`, z tends to be small, i.e., close to zero and in this range, say $(-\varepsilon, \varepsilon)$ the model tends towards being linear. 

- **(Inverted) Dropout Implementation**:

Suppose I implement the dropout regularization on a particular layer with `keep_prob = 0.8`. Then on average, the output `Z = W.A + b` would reduce by 20%, causing an overall reduction on the output values. In order to correct this, we divide the layer by `keep_prob`, i.e., `A = A / keep_dims`.  

- It can be shown that dropout has a similar effect as L2 regularization, i.e., shrink weights.

- Usually input layers have `keep_prob = 1`. Typically use high values of `keep_prob` to layers where you might think overfit more. This means more hyperparameters to tune over. If that isn't preferable, use dropout on select layers _only_ and use the same `keep_prob` in each. 

- Note: Since the cost function `J` with dropout is not well-defined, when using the error plots, we ignore the dropout.

- **Other Regularizations**: 
    - Data augumentations (If detecting images, augument to training set by modifying existing images - distortion, rotation). [Video link](https://www.coursera.org/learn/deep-neural-network/lecture/Pa53F/other-regularization-methods)

    - Early stopping (Stop training the neural network when holdout set error starts increasing). [Same video]

- **Orthogonalization**: 
    - Concentrate on one task at a time, minimize J(W, b) first and not worry about overfitting, then do regularization. [Same video]

- **Normalization**: Normalize the test set with the same mean/standard deviation as that of the training set! Sometimes when inputs are in, say (0, 1) and (-1, 1) and (2, 3) then normalization is **not** necessary. [Video link](https://www.coursera.org/learn/deep-neural-network/lecture/lXv6U/normalizing-inputs) 

- Vanishing/Exploding gradients: To solve this problem, use variants of Keras' `lecun_uniform` random initialization. [Video reference](https://www.coursera.org/learn/deep-neural-network/lecture/C9iQO/vanishing-exploding-gradients)


