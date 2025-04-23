# Temparature
Temperature controls the degree of randomness in token selection. Lower temperatures are good for prompts that expect a more deterministic response, while higher temperatures
can lead to more diverse or unexpected results. A temperature of 0 (greedy decoding) is deterministic: the highest probability token is always selected (though note that if two tokens
have the same highest predicted probability, depending on how tiebreaking is implemented you may not always get the same output with temperature 0). 

Temperatures close to the max tend to create more random output. And as temperature gets higher and higher, all tokens become equally likely to be the next predicted token.

## How It Works (Conceptually)
1. The model generates a probability distribution over all possible next tokens (say 50,000+).
2. These probabilities are logits (raw scores) which are then softmaxed into a probability distribution.
3. Temperature is applied before softmax like this:

```python
adjusted_logits = logits / temperature
probs = softmax(adjusted_logits)
```
So, the temperature directly scales the logits before they go through softmax.


## Examples

### Temparature = 0

|             |                                                                                       | 
|-------------|---------------------------------------------------------------------------------------|
| Name        | 01_temp_0                                                                             |
| Model       | Gemma3:12b                                                                            |
| Temperature | 0                                                                                     |
| Token Limit | -                                                                                     |
| Top-K       | -                                                                                     |
| Top-P       | -                                                                                     |
| Prompt      | what is the capital of germany?                                                       |
| Output      | "The capital of Germany is **Berlin**.\n\n\n\nIt's also the largest city in Germany!" |

curl: 
```shell
curl --request POST --no-buffer -sS --url http://localhost:11434/api/generate --header 'content-type: application/json' --data '{
    "model": "gemma3:12b",
    "prompt": "what is the capital of germany?",
    "stream": false,
    "options": {
      "temperature": 0
    }
}' | jq .response
```

This response does not change with different runs. The model always returns the same output. This is because the temperature is set to 0, which means that the model will always select the token with the highest probability.

### Temparature = 1

#### Example 1
|             |                                                                                                  | 
|-------------|--------------------------------------------------------------------------------------------------|
| Name        | 01_temp_1_1                                                                                      |
| Model       | Gemma3:12b                                                                                       |
| Temperature | 1                                                                                                |
| Token Limit | -                                                                                                |
| Top-K       | -                                                                                                |
| Top-P       | -                                                                                                |
| Prompt      | what is the capital of germany?                                                                  |
| Output      | "The capital of Germany is **Berlin**.\n\n\n\nIt's a vibrant and historically significant city!" |


#### Example 2
|             |                                                                                       | 
|-------------|---------------------------------------------------------------------------------------|
| Name        | 01_temp_1_2                                                                           |
| Model       | Gemma3:12b                                                                            |
| Temperature | 1                                                                                     |
| Token Limit | -                                                                                     |
| Top-K       | -                                                                                     |
| Top-P       | -                                                                                     |
| Prompt      | what is the capital of germany?                                                       |
| Output      | "The capital of Germany is **Berlin**.\n\n\n\nIt's also the largest city in Germany!" |


curl:
```shell
curl --request POST --no-buffer -sS --url http://localhost:11434/api/generate --header 'content-type: application/json' --data '{
    "model": "gemma3:12b",
    "prompt": "what is the capital of germany?",
    "stream": false,
    "options": {
      "temperature": 1
    }
}' | jq .response
```

This response changes with different runs. The model returns different outputs each time. This is because the temperature is set to 1, which means that the model will select the next token based on a probability distribution. The higher the temperature, the more random the output will be.

### Temparature = 10 (unhinged mode)
#### Example 1
|             |                                                                                                                                                                                                                                                                                                                | 
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Name        | 01_temp_10_1                                                                                                                                                                                                                                                                                                   |
| Model       | Gemma3:12b                                                                                                                                                                                                                                                                                                     |
| Temperature | 10                                                                                                                                                                                                                                                                                                             |
| Token Limit | -                                                                                                                                                                                                                                                                                                              |
| Top-K       | -                                                                                                                                                                                                                                                                                                              |
| Top-P       | -                                                                                                                                                                                                                                                                                                              |
| Prompt      | what is the capital of germany?                                                                                                                                                                                                                                                                                |
| Output      | "The capiasljyjt ofr manywgtkjymutygbgGermany rkmjnklln hbt isvBeredllijnBerlin.!\n Berlin bjtjnffgbbnmnnm\n\n\nHopefully my mistake amusedyouu\n\n\nAnyway as well ,\nAs Berlin as vggttty the federal nngtyjm y gjfgn fg hjk government; therefore the fgjjgtty ybtt gggtty is B errdssj j t berlinhjgn n\n" |

#### Example 2
|             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Name        | 01_temp_10_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Model       | Gemma3:12b                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Temperature | 10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Token Limit | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Top-K       | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Top-P       | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Prompt      | what is the capital of germany?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Output      | "the is technically nuanced; west germany historically declared BONNS until there Germany reunies under what united the federal states which formally renamed Bern' to it official of German in in modern unified form.. Today. As result official governing official legal terms there official it not legally has official designated; this Germany the\n\nas far official current Germany' s place place official Germany today capital formally! so location place which that' be there place, . that a government Germany location located near east-\n\na\n\n" |

curl:
```shell
curl --request POST --no-buffer -sS --url http://localhost:11434/api/generate --header 'content-type: application/json' --data '{
    "model": "gemma3:12b",
    "prompt": "what is the capital of germany?",
    "stream": false,
    "options": {
      "temperature": 10
    }
}' | jq .response
```

This meaningless word salads are the result of setting the temperature to 10. The model is now in "unhinged mode", where it generates random and nonsensical text. This is because the temperature is set to a very high value, which means that the model will select the next token based on a very flat probability distribution. 
which means that all tokens are equally likely to be selected. This results in a completely random output that does not make any sense.

