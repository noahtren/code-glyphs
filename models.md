# Language Models

HuggingFace codeBERTa is ~83 million parameters.
We have one as a language encoder and one as a
language decoder.

BiT ResNet50x1 is ~23 million parameters.

Generator is 64 million parameters with 3 generator levels.
Generator is 66 million parameters with 5 generator levels.

TODO: improve generator architecture, ideally borrowing from a pretrained
models or elegant generator design.

(83 * 2) ~vision~ + 68 ~generator~ + 23 ~disc~ = 257 million params,
or ~1 billion bytes. Understandable that the models weigh so much.
