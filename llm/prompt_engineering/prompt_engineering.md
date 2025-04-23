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
- **Top-P (nucleus sampling)**: sampling selects the top tokens whose cumulative probability does not exceed a certain value (P). Values for P range from 0 (greedy decoding) to 1 (all tokens in the LLM’s vocabulary).
    See [Top-P](./02_top_p.ipynb) for more details and examples.
- **Top-K**: sampling selects the top K most likely tokens from the model’s predicted distribution. The higher top-K, the more creative and varied the model’s output; the lower top-K, the more restive and factual the model’s output. A top-K of 1 is equivalent to greedy decoding.
    See [Top-K](./03_top_k.ipynb) for more details and examples.
- **Output length**: The maximum number of tokens to generate in the output. This can be set to a specific value or left to the model to determine based on the input prompt. Reducing the output length of the LLM doesn’t cause the LLM to become more stylistically
  or textually succinct in the output it creates, it just causes the LLM to stop predicting more tokens once the limit is reached. If your needs require a short output length, you’ll also possibly need to engineer your prompt to accommodate.
   See [Output length](./04_output_length.ipynb) for more details and examples.


