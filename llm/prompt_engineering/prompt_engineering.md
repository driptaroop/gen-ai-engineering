# Prompt Engineering

## What is prompt engineering?
Prompt engineering is the process of designing and refining prompts to elicit the desired response from a language model. 
It involves understanding how the model interprets different types of input and crafting prompts that guide the model towards generating relevant and accurate outputs.

## Tuning LLMs with sampling controls
Tuning prompts is one of the ways to improve the usefulness of LLMs responses. But it is not the only way.
Another way is to tune the model response via sampling controls. LLMs do not formally predict a single token. Rather, LLMs predict probabilities for what the
next token could be, with each token in the LLM’s vocabulary getting a probability. Those token probabilities are then sampled to determine what the next produced token will be.

There are several parameters that can be adjusted to influence the sampling process and the resulting output.

- **Temperature**: Controls the randomness of the model's output. A lower temperature (e.g., 0.2) makes the output more deterministic, while a higher temperature (e.g., 0.8) increases randomness and creativity.
    See [Temperature](./01_temparature.ipynb) for more details and examples.



