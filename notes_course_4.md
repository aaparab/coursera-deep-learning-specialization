---
# Course 4. Convolutional Neural Networks
---

## Week 1: Foundations of Convolutional Neural Networks

- **Valid convolution** transforms an nxn image convolved with an fxf filter into an (n-f+1)x(n-f+1) image. No padding edges. [Video reference](https://www.coursera.org/learn/convolutional-neural-networks/lecture/o7CWi/padding)  

- **Same convolution** pads the input image by a pxp border so that the output has same dimensions as the input. The condition is

    (n + 2p - f + 1) x (n + 2p -f + 1) = n x n ==> f = 2p + 1.

- **Strided convolution** with a stride of `s` gives an output matrix of size 

   floor((n+2p-f)/s + 1) x floor((n+2p-f)/s + 1). 

- **3D convolutions**: Convolving a (n x n x n_c) image with n_c' filters of size (f x f x n_c) gives an image of size (n - f + 1 x n - f + 1 x n_c'). 

- **Max Pooling** rarely uses padding. The formula for computing the output matrix dimensions is the same as that of `same` convolutions. 

- Typical examples of CNNs use the following layers:

    INPUT - CONV1 - POOL1 - CONV2 - POOL2 - FC - FC - FC - SOFTMAX

- Also typically the dimensions of each matrix reduce from left (input) layer to right (output) whereas the number of filters increases. 

- **Parameter sharing**: A feature detector (such as a vertical edge detector) that is useful in one part of the image is probably useful in another part of the image. 

- **Sparsity of connections**: In each layer, each output value depends only on a small number of inputs. 

- The convolutional structure makes the image detection _translation-invariant_. 

