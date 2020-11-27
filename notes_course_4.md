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

## Week 2: Deep convolutional models: case studies

- LeNet-5: [Wiki](https://en.wikipedia.org/wiki/LeNet#Structure[5]_[6])

    - Around 60K parameters.
    - Filter dimensions n_H, n_W decrease to right.
    - number of channels n_C increases to right.
    - Typical arrangements: CONV --> POOL --> CONV --> POOL --> FC --> FC --> OUTPUT
    - Used _sigmoid, tanh_ not relu back then. 
    - [Original paper](https://pdfs.semanticscholar.org/62d7/9ced441a6c78dfd161fb472c5769791192f6.pdf)

- AlexNet: [Wiki](https://en.wikipedia.org/wiki/AlexNet#Network_design)

    - [Original paper](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
    - 60M parameters. 
    - Used _relu_ activation, _dropout_ regularization, multiple GPUs. 
    - Easy paper to read, says Ng. 

- VGG-16:

    - [Original paper](https://arxiv.org/abs/1409.1556)

- ResNet: [Wiki](https://en.wikipedia.org/wiki/Residual_neural_network)
    
    - [Original paper](https://arxiv.org/abs/1512.03385)
    - Information from layer l goes through a _short cut_ (or _skipped connection_) to layer (l+2)
    - a^[l+2] = g(z^[l+2] + a^[l])
    - [Video reference](https://www.coursera.org/learn/convolutional-neural-networks/lecture/HAhz9/resnets) 
    - Used to alleviate vanishing and exploding gradients problem
    - What goes wrong in very deep plain nets is that when you make the network deeper and deeper, it is actually very difficult for it to choose parameters that learn even the identity function, which is why a lot of layers end up making the result worse. 
    - The main reason that residual network works is that it is so easy for these extra layers to learn the identity function that you are kind of guaranteed that it doesn't hurt performance and then a lot of time you get lucky and even helps performance. [Video reference](https://www.coursera.org/learn/convolutional-neural-networks/lecture/XAKNO/why-resnets-work)

- 1x1 convolution: Basically having a fully-connected network that multiplies each channel and outputs one node. Also called _network-in-network_. 

    - This is a way to shrink n_C. Use n_C filters of size 1x1xn_C_prev. 

- Inception network:

    - Concatenates 1x1, 3x3, 5x5 filters and MAXPOOL layers into one Inception layer. 
    - ![We need to go deeper](https://miro.medium.com/max/1400/0*W8LNnUr9FZLH7ghg.jpg)
    - Softmax layer attached to intermediate hidden layers also seems to give reasonable prediction. This appears to have a regularizing effect. 

- Using open-source implementation: Always a good idea to use models and weights trained by other open-source implementations. 

- Transfer Learning: [Video reference](https://www.coursera.org/learn/convolutional-neural-networks/lecture/4THzO/transfer-learning) 
    - If you have less data, use the entire model and weights of an open-source trained model and just change the last (softmax) layer to suit your application. 
    - If you have moderate sized data, freeze (set as non-trainable) the first few layers and train the rest. 

    - If you have a huge dataset, keep all layers trainable but only use the trained weights as the initial state of the model. 

- Data augumentation: More often than not, computer vision problems can be solved by having enough data. So one can create more training examples by techniques such as **mirroring**, **random cropping**, **rotation**, **shearing** and **color shifting** (+20R, -20G, +10B) etc. [Video reference](https://www.coursera.org/learn/convolutional-neural-networks/lecture/AYzbX/data-augmentation) 

    - The AlexNet paper uses _PCA color augumentation_. 


## Week 3: Detection algorithms

- Object localization: Along with the output predicting a bunch of objects, say pedestrian/car/motorcycle/light using a softmax layer, add output vector that also predicts the coordinates of the bounding box. Note that the training set needs to have these coordinates also. Moreover the loss function needs to be modified appropriately, see [Video reference](https://www.coursera.org/learn/convolutional-neural-networks/lecture/nEeJM/object-localization). 

- Landmark detection: Along with the ConvNet output predicting an object, add say 64 coordinates of the object we are interested in. [Video reference](https://www.coursera.org/learn/convolutional-neural-networks/lecture/OkD3X/landmark-detection). 

- Sliding window detection: Once a ConvNet is trained to detect an object, crop the image under consideration and pass each cropped image through the CNN. Change window size. [Video reference](https://www.coursera.org/learn/convolutional-neural-networks/lecture/VgyWR/object-detection). 

- Convolutional Implementation of Sliding Window: There is a lot of redundency in the previous sliding window construction. That can be avoided by using a convolutional implementation of a sliding window. [Video reference](https://www.coursera.org/learn/convolutional-neural-networks/lecture/6UnU4/convolutional-implementation-of-sliding-windows). 

- Advantage of convolutional implementation of sliding window: For chessboard piece classification!

- YOLO algorithm for object-detection: [Original paper](https://arxiv.org/pdf/1506.02640.pdf) Ng says difficult to read. 

- Intersection over Union (IOU): Take ratio of intersection to union of the two sets - bounding box containing object, and grid (filter) box. If this ratio is > threshhold, then `correct`. This ratio is 1 if and only if the two boxes coincide. 

- Non-max suppression algorithm: [Video reference](https://www.coursera.org/learn/convolutional-neural-networks/lecture/dvrjH/non-max-suppression) 

    - For each box, we predict a vector containing
        - the probability of the object being in the box (p_c)
        - the centroid and coordinates of the box, and
        - softmax identifying the object between few classes. 
    - Discard all boxes with p_c < 0.6
    - Pick the box with the largest p_c, output that as a prediction
    - Discard any remaining box with IoU >= 0.5 with the box output in the above step. 
    - Repeat past 2 steps for each remaining box. 

- Anchor boxes to identify overlapping objects: To detect two objects, use multiple vectors (called anchor boxes) like above. Assign an object to that anchor box which has the maximum IoU. 


