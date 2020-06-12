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

  prompt = "def sum(a, "
  prompt_ids = tokenizer.encode(prompt, return_tensors='tf')
  result = model.generate(
    prompt_ids,
    do_sample=True,
    max_length=50,
    top_k=5
  )
  print(tokenizer.decode(result[0]))
