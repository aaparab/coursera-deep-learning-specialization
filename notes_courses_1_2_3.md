---
# Course 1. Neural Networks and Deep Learning
---

## Week 1: Introduction to deep learning

- Scale drives deep learning progress: Increase in labelled data, computational power and better algorithms have made deep learning take off. 

- The slope of the sigmoid function is close to zero for really small and really large input values. This makes relu a better choice, since **gradient descent becomes faster**.[Video reference](https://www.coursera.org/learn/neural-networks-deep-learning/lecture/praGm/why-is-deep-learning-taking-off)

## Week 2: Neural network basics

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


---
# Course 2. Improving Deep Neural Networks
---

## Week 1: Practical aspects of Deep Learning

#### Much of the stuff in this week is discussed in Yann Lecun's excellent paper [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf). Must read! 

|               |     Model    | Train error | Test error | How to resolve                                       |
|:-------------:|:------------:|:-----------:|:----------:|------------------------------------------------------|
|   High Bias   | Underfitting |     15%     |     16%    | Bigger network, Train longer, Change hyperparameters |
| High Variance |  Overfitting |      1%     |     11%    | More data, Regularization                            |
|      Both     |              |     15%     |     30%    |                                                      |
|    Neither    |              |      1%     |     2%     |                                                      |

This is assuming very low Bayes error, i.e., with best available or human score.Otherwise consider (Train error - Bayes error).

Assumption: Both training set and test set are from the same distribution.

- **Bias-Variance tradeoff**: Used to exist in pre-Deep Learning days. But now, using bigger network and getting more data can help in reducing both - bias and variance. So no longer a trade-off. One reason why Deep Learning has taken off for supervised learning. [Video reference](https://www.coursera.org/learn/deep-neural-network/lecture/ZBkx4/basic-recipe-for-machine-learning)

- Why not regularize the bias term? In practice there are many input parameters and hidden nodes so not adding the regularization to _one_ of the terms (intercept `b`) shouldn't affect the overall training. [Video reference](https://www.coursera.org/learn/deep-neural-network/lecture/Srsrc/regularization) 

- L^1 v/s L^2 regularization: Using L^1 regularization makes the weight vector sparse. (Why?) [Same video as above]

- **L^2 regularization is Weight Decay**: The expression for updating the weight `w` is 
```
w<sup>l</sup> := w<sup>l</sup> - &alpha;*&lambda;/m w<sup>l</sup> - &alpha; (from BackProp)
```
This reduction in `w<sup>l</sup>` is referred to as *weight decay*.

- **Regularization prevents overfitting (Intuition ONLY)**: 
    1. If lambda > 0 then weights w tend to be small, which is equivalent to turning off many neurons, which makes the model closer to being linear. 
    2. If lambda > 0 then weights w tend to be small, so in the equation 
        `z = w*a + b`, z tends to be small, i.e., close to zero and in this range, say $(-&epsilon;, &epsilon;)$ the model tends towards being linear.

- **(Inverted) Dropout Implementation**:

Suppose I implement the dropout regularization on a particular layer with `keep_prob = 0.8`. Then on average, the output `Z = W.A + b` would reduce by 20%, causing an overall reduction on the output values. In order to correct this, we divide the layer by `keep_prob`, i.e., `A = A / keep_dims`.  

- It can be shown that dropout has a similar effect as L2 regularization, i.e., shrink weights.

- Usually input layers have `keep_prob = 1`. Typically use high values of `keep_prob` to layers where you might think overfit more. This means more hyperparameters to tune over. If that isn't preferable, use dropout on select layers _only_ and use the same `keep_prob` in each. 

- Note: Since the cost function `J` with dropout is not well-defined, <span style="text-decoration: underline">when using the error plots, we ignore the dropout.<\span>

- **Other Regularizations**: 
    - Data augumentations (If detecting images, augument to training set by modifying existing images - distortion, rotation). [Video link](https://www.coursera.org/learn/deep-neural-network/lecture/Pa53F/other-regularization-methods)

    - Early stopping (Stop training the neural network when holdout set error starts increasing). [Same video]

- **Orthogonalization**: 
    - Concentrate on one task at a time, minimize J(W, b) first and not worry about overfitting, then do regularization. [Same video]

- **Normalization**: Normalize the test set with the same mean/standard deviation as that of the training set! Sometimes when inputs are in, say (0, 1) and (-1, 1) and (2, 3) then normalization is **not** necessary. [Video link](https://www.coursera.org/learn/deep-neural-network/lecture/lXv6U/normalizing-inputs) 

- Vanishing/Exploding gradients: To solve this problem, use variants of Keras' `lecun_uniform` random initialization. [Video reference](https://www.coursera.org/learn/deep-neural-network/lecture/C9iQO/vanishing-exploding-gradients)

## Week 2: Optimization algorithms

- With mini-batch gradient descent, typical batch size: 64, 128, 256, 512. [Video reference](https://www.coursera.org/learn/deep-neural-network/lecture/lBXu8/understanding-mini-batch-gradient-descent)

- Momentum: 
```
To compute an exponentially weighted average of the gradients, and then use that gradient to update the weights.
```

- **Adam = Gradescent with Momentum + RMSProp**.

- Learning rate decay: Slowly reduce learning rate over time.

## Week 3: Hyperparameter Tuning, Batch Normalization and Programming Frameworks

- Most important hyperparameter to tune: Learning rate `alpha`. 

- Next in importance: `beta` in momentum / RMSProp, # hidden units, mini-batch size.

- Main steps: Use random values and **not** grid search; do coarse-to-fine. [Video reference](https://www.coursera.org/learn/deep-neural-network/lecture/dknSn/tuning-process)

- Intuitions do get stale. Re-evaluate (hyperparameters) occasionally. 

### Batch Normalization

- Can we normalize z<sup>[l]</sup> (or a<sup>[l]</sup>) so as to train the next layer w<sup>[l+1]</sup>, b<sup>[l+1]</sup> faster? [Video link](https://www.coursera.org/learn/deep-neural-network/lecture/4ptp2/normalizing-activations-in-a-network); also see [Keras](https://keras.io/api/layers/normalization_layers/batch_normalization/) 

- [Original paper](https://arxiv.org/pdf/1502.03167.pdf): Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. 

- Why batch normalization works: [Video reference](https://www.coursera.org/learn/deep-neural-network/lecture/81oTm/why-does-batch-norm-work)
    - Normalizing the input features speeds up learning
    - It makes weights deeper in your network, say weight on layer 10, more robust to changes in weights in earlier layers of the network. 

- Covariate shift: Data distribution changing while training (say after changing the mini-batch).

- What batch normalization does, is that it limits the amount to which updating the parameters in the early layers can affect the distribution of balues that the latter layer sees and therefore has to learn on.

- Has some regularization effect: The batch normalization adds some noise to the values within that minibatch similar to dropout, making dropout slightly redundent. So also increasing the batch size reduces (the noise and hence, reduces) this regularization effect.



- Multiclass classification: Softmax. Generalization of logistic regression; same loss function as LR, i.e., L(y, \hat{y}) = \sum_{j = 1}^c y_j log(\hat{y_j}). 


---
# Course 3: Structuring Machine Learning Projects
---

## Week 1: Introduction to ML strategy

- Satisfying/Optimizing metric: If you have `N` metrics, Ng recommends having `N-1` satisfying metrics (e.g., run time < 100ms) and `1` optimizing metric. [Video reference](https://www.coursera.org/learn/machine-learning-projects/lecture/uNWnZ/satisficing-and-optimizing-metric)

- Train/dev/test ratio: Choose dev and test set size to be the smallest size that gives high confidence in the overall performance of the system. 
    - Previously: 70/30
    - Nowadays: 98/1/1

- **Avoidable bias** = Training error - Human-level (Bayes') error

  **Variance** = Dev error - Training error 

## Week 2: Error Analysis

- Counting the fraction of mislabelled images gives an idea about which direction to proceed. [Video reference](https://www.coursera.org/learn/machine-learning-projects/lecture/IGRRb/cleaning-up-incorrectly-labeled-data) 

- DL algorithms are quite robust to random errors in the training set. Thus incorrectly labelled examples are okay so long as their percentage is small. These mislabellings should **not** be systematic errors. 

- Ng's strategy of error analysis: Take 100 mislabelled examples and tabulate them with common reasons of being mislabelled. For instance, 

| Image | Dog | Big cat | Mislabelled | Remarks |
|-------|-----|---------|-------------|---------|
| 1     | Y   |         |             | ..      |
| 2     |     |         | Y           |         |
| .     |     |         |             |         |
| .     |     |         |             |         |
| Total | 12  | 30      | 58          |         |

- Training and testing on different distributions: 
    He discusses what to do when the test-quality images are few and training-quality images are more. Put all training-quality images in the training set and the dev and test sets should comprise of the **same** test-quality images. [Video reference](https://www.coursera.org/learn/machine-learning-projects/lecture/Xs9IV/training-and-testing-on-different-distributions)

- Bias and variance with mismatched data distributions:
    In the above problem, it is difficult to quantify that the observed variance was due to overfitting on the training set, or due to the dev set being _inherently different_. In this case, sub-divide the entire training set into a training set (where to use backprop) and another training-dev set to make inferences. Note that

    - Train and train-dev have same distribution
    - dev and test have the same distribution. 

    Doing so, we realize that variance is the difference between training and training-dev set. Thus we can identify between **variance** problem and **data-distribution** problem. 

| ERROR ANALYSIS      | %   | Diff | Cause                            |
|---------------------|-----|------|----------------------------------|
| Human (Bayes) error | 4%  |      |                                  |
|                     |     | 3%   | Avoidable bias                   |
| Training set error  | 7%  |      |                                  |
|                     |     | 3%   | Variance                         |
| Training-dev error  | 10% |      |                                  |
|                     |     | 2%   | Data mismatch                    |
| Dev error           | 12% |      |                                  |
|                     |     | 0%   | Degree of overfitting to dev set |
| Test error          | 12% |      |                                  |

- Transfer Learning: Task A &rarr; Task B
    
    - Task A and B have the same input x,
    - Lot more data for task A than B,
    - Low-level features from A could be helpful for learning B. 

- Multi-task Learning: 
    
    - Training on a set of tasks that could benefit from having shared low-level features (e.g., detect stop sign, pedestrian, lights in autonomous driving).
    - Data for each task is quite similar. 


