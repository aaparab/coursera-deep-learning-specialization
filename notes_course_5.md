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



## Week 2: Natural Language Processing & Word Embeddings

- Transfer learning and word embeddings:
    
    - Learn (download) word embeddings from large text corpus (1-100B words)

    - Transfer embeddings to new task with small training set (100k words)

    - (Optional) Continue to finetune the word emeddings with new data.

- Word embeddings make precise the following analogy:

    - man : woman :: king : ?

- The similarity function most commonly used is the `cosine similarity`: 

    - sim(e_w1, e_w2) = \frac{e_w1^T . e_w2}{ ||e_w1||_2 .  ||e_w2||_2 }. 

- Can also use `norm` as a dissimilarity function. 

- **Embedding matrix E**: It is a matrix of dimensions mxn where m is the dimension of the word-to-vector space and n is the size of the corpus. The j-th column of E represents the vector corresponding to the j-th word in the corpus. [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/K604Z/embedding-matrix) 

- The matrix E can be learned by using gradient descent! [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/K604Z/embedding-matrix) 

## Week 3: Sequence models & Attention mechanism

### Sequence to sequence architectures:

- [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/v2pRn/picking-the-most-likely-sentence) For tasks such as image captioning or translating a sentence, the usual method of inputting a sequence x^<1>, ..., x^<T_x> to generate an output y^<1>, ..., y^<T_y> needs to be modified. If x^<1>, ..., x^<T_x> represents an encoding of the given input, we feed this input to the RNN first and only then the RNN makes a prediction of the output sequence y^<1>, ..., y^<T_y>. 

- Consider two English translations of the French sentence 
```
Jane visite l'Afrique en septembre. 
```
namely
```
Jane is visiting Africa in September. 
```
and 
```
Jane is going to be visiting Africa in September. 
```
This is a case when a greedy algorithm does not necessarily work, i.e., since `going` is a more popular English word, it might be that 
```
P("Jane is going" | "Jane visite l'Afrique en septembre.") > P("Jane is visiting" | "Jane visite l'Afrique en septembre.")
```
Hence the *Beam Search* algorithm below. [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/4EtHZ/beam-search)

- Two extreme approaches: Greedy means we only consider the most likely word as the next occurring one. On the other hand considering the entire corpus with their probability gets out of hand in a few words. (Think of the number of ways of playing the first 10 chess moves). 

- *Beam search*: Instead of considering the entire corpus, only consider the top `beam_width` possibilities at every step. At every step we instantiate `beam_width` copies of the network to evaluate partial sentence fragments. 

- `beam_width` = 1 reduces to the greedy search algorithm. 

- Refinement to beam search: Since product of probabilities tends to be numerically unstable, consider applying logarithm and normalizing (by T_y or by T_y^\alpha with \alpha \in (0, 1)). 

- Error analysis with beam search: Whether beam search or the inherent RNN is responsible for poor performance? 

    Given the best sentence-translation, say human-generated called as y*, we compare the probabilities P(y* | x) and P(\hat{y} | x). 

    - If P(y* | x) > P(\hat{y} | x): Beam search chose \hat{y} instead of y* so it is at fault. 
    - If P(y* | x) <= P(\hat{y} | x): RNN model couldn't predict the best translation. 

    Aggregate this for a set of dev set examples. 

### Attention

- Given a very long sentence, it is difficult to translate the entire sentence so a human would break it down into phrases and try to translate each part. 

- Analogously the RNN marches forward generating one word at a time, until eventually it generates the EOS, and at every step there are attention weights. [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/RDXpX/attention-model-intuition)

- More details about attention: [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/lSwVa/attention-model) Look at context vectors. 

### CTC

- Speech recognition: End-to-end speech recognition has made obsolete the conventional idea of translating audio speech into [phonemes](https://en.wikipedia.org/wiki/Phoneme) to convert into text. 

- CTC cost for speech recognition ([Connectionist temporal classification](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.75.6306)) Although typically T_x is large for audio inputs and T_y (sentence) is small, we use T_x = T_y. The basic rule is to introduce a "blank" character and collapse repeated characters not separated by "blank". [Video reference](https://www.coursera.org/learn/nlp-sequence-models/lecture/sjiUm/speech-recognition)
