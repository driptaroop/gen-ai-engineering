# Foundational LLMs and Text Generation

> Based on whitepaper: https://www.kaggle.com/whitepaper-foundational-llm-and-text-generation

## Introduction
Advent of LLMs(Large Language Models) has revolutionized the world of Artificial Intelligence. Their ability to process, 
generate, and understand user intent is fundamentally changing the way we interact with information and technology.

## Table of Contents
- Building Blocks of LLMs
- Transformer Architecture
- Attention Mechanism
- LLMs such as BERT, GPT, Gemini
- Training and Fine-tuning
- Improve and Optimize response generation

## Large Language Models
A language model predicts the probability of a sequence of words. For example, given the prefix “The most famous city in the US is…”, a language model might predict high
probabilities to the words “New York” and “Los Angeles” and low probabilities to the words “laptop” or “apple”.

Before the invention of transformers, popular approach of languages models were RNNs (Recurrent Neural Networks). Particularly, LSTMs (Long Short Term Memory) 
and GRUs (Gated Recurrent Units) were used. These models are sequential in nature, meaning they process one word at a time, which makes them slow and inefficient for long sequences.
The sequential nature of RNNs make them compute intensive and difficult to parallelize during training.

Transformers, on the other hand, are a type of neural network that can process sequences of tokens in parallel thanks to the self-attention mechanism.
This allows them to capture long-range dependencies and relationships between words more effectively than RNNs. 
However, transformers are still limited by the amount of data they can process at once, which is typically constrained by the size of the model and the available computational resources,
while RNNs theoretically have infinite context length. Although, in practice, RNNs are limited by the vanishing gradient problem(todo: what is it?), which makes it difficult for them to learn long-range dependencies.

## Transformer Architecture
The transformer architecture was developed at Google in 2017 for use in a translation model. It’s a sequence-to-sequence model capable of converting sequences from one domain into sequences in another domain.
For example, it can convert a sequence of words in English into a sequence of words in French. 
The original transfer architecture consisted of 2 parts, the encoder and the decoder. The encoder takes a sequence of words in the source language and converts it into a representation of the sequence in a continuous vector space.
The decoder takes this representation and converts it into a sequence of words in the target language.

![img.png](img.png)

The transformer consists of multiple layers. A layer in a neural network comprises a set of parameters that perform a specific transformation on the data. In the diagram you can see an example of some layers which include Multi-Head Attention, Add & Norm, Feed-Forward,
Linear, Softmax etc. The layers can be sub-divided into the input, hidden and output layers. The input layer (e.g., Input/Output Embedding) is the layer where the raw data enters the
network. Input embeddings are used to represent the input tokens to the model. Output embeddings are used to represent the output tokens that the model predicts. For example, in
a machine translation model, the input embeddings would represent the words in the source language, while the output embeddings would represent the words in the target language.
The output layer (e.g., Softmax) is the final layer that produces the output of the network. The hidden layers (e.g., Multi-Head Attention) are between the input and output layers and are where the magic happens!

