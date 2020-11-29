---
# Course 5: Sequence Models
---

## Week 1: Recurrent Neural Networks

- Basics of RNN: When you make prediction for y^<t>, we not only consider x^<t> but also information from x^<t-1>, x^<t-2>, ... . 

- One limitation of RNNs is that the predictino at a certain time uses inputs or information from the inputs earlier in the sequence but not information later in the sequence. 

- Equations:
    a^<t> = g_1(W_a [ a^<t-1>, x^<t> ] + b_a)
    y^<t> = g_2(W_y a^<t> + b_y)

    Where W_a is a concatenated matrix corresponding to [W_{a, a} | W_{a, x}]. 

- Typical activations: `tanh`, `relu` for a and `sigmoid` for y. 

- Backpropagation through time [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/bc7ED/backpropagation-through-time) 

- Types of RNNs: 

    - One to one
    - One to many
    - Many to one
    - Many to many
    - Many to many (encoder followed by decoder)

    See Andrej Karpathy's [blog](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) titled "The unreasonable effectiveness of RNNs". [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/BO8PS/different-types-of-rnns)

- Sampling novel sequences: Can use random starting x^<0> to generate sequences. [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/MACos/sampling-novel-sequences)

- Vanishing gradients: The sentence below,
    
    `The cat(cats), which already ate a hearty meal, was(were) full.`
  
    needs a `was` or `were` depending on whether the much earlier word is `cat` or `cats`. A long sequence needs a deeper network and this causes the _vanishing gradients_ problem. [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/PKMRR/vanishing-gradients-with-rnns)

- The problem of exploding gradients can be solved using [gradient clipping](https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48), i.e., controlling the norm of the gradient vector. 

- **GRU** (Gated Recurrent Units): This is a generalization of RNNs. When Gamma_u = 1 we recover the RNN architecture. We have a memory cell `c = c^<t>` which we update or not, depending on the Gate `Gamma_u`. In the special case of RNNs, we have `c = a`. Equations:

    ```
    \tilde{c}^<t> = tanh(W_c [\Gamma_r * c^<t-1>, x^<t>] + b_c)
    \Gamma_u = \sigma(W_u [c^<t-1>, x^<t>] + b_u)
    \Gamma_r = \sigma(W_r[c^<t-1>, x^<t>] + b_r)
    c^<t> = \Gamma_u * \tilde{c}^<t> + (1-\Gamma_u) * c^<t-1>
    ```

![RNN v/s LSTM v/s GRU](http://dprogrammer.org/wp-content/uploads/2019/04/RNN-vs-LSTM-vs-GRU.png)

- **LSTM** (Long Short Term Memory): More powerful than GRUs. It has three gates, namely \Gamma_u, \Gamma_f (forget) and \Gamma_o (output). The equations are below:

    ```
    \tilde{c}^<t> = tanh(W_c [c^<t-1>, x^<t>] + b_c)
    \Gamma_u = \sigma(W_u [c^<t-1>, x^<t>] + b_u)
    \Gamma_f = \sigma(W_f[c^<t-1>, x^<t>] + b_f)
    \Gamma_o = \sigma(W_o[c^<t-1>, x^<t>] + b_o)
    c^<t> = \Gamma_u * \tilde{c}^<t> + \Gamma_f * c^<t-1>
    a^<t> = \Gamma_o * tanh(c^<t>).
    ```

So whereas we had \Gamma and (1-\Gamma) in GRUs, here we have the option of keeping the old value c^<t-1> and adding the updated value \tilde{c}^<t>. 

- [Chris Olah blog post on LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- Sometimes there is a _peephole connection_ which is a variation of LSTMs. 

- Computationally GRUs are better (less complicated so easier to scale) but LSTMs are more powerful and effective. If you had to pick one, LSTM has been the default choice. 

- Bidirectional RNNs: Blocks going from right to left. [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/fyXnn/bidirectional-rnn) 

- Deep RNNs: Because of the temporal dimension, these networks can already get quite big even if you have just a small handful of layers. However sometimes we have a deep neural network following the output of the RNN.


