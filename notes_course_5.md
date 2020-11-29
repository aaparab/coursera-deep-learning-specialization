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

    See Andrei Karpathy's [blog](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) titled "The unreasonable effectiveness of RNNs". [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/BO8PS/different-types-of-rnns)


