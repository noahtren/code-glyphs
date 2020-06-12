from transformers import pipeline

from language import get_transformer, tokenizer

if __name__ == "__main__":
  model = get_transformer(LM=True)
  fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
  )
  example = "for i in <mask>(10): print(i)"
  print(fill_mask(example))
