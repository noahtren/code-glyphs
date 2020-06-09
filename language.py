"""Encode/decode source code to/from a fixed-length representation.

Unfortunately not using CODEBert because it is only available in PyTorch.
Instead, using the generic distilBERT language model.
"""

import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer, TFDistilBertForMaskedLM

from cfg import get_config; CFG = get_config()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')


def get_transformer(LM:bool):
  if LM:
    model = TFDistilBertForMaskedLM.from_pretrained(
      'distilbert-base-cased'
    )
  else:
    model = TFDistilBertModel.from_pretrained(
      'distilbert-base-cased',
    )
  return model


def batch_encode(raw_texts:list):
  flatten_output = False
  if isinstance(raw_texts, tf.Tensor):
    flatten_output = True
    raw_texts = [tf.compat.as_str(raw_texts.numpy())]
  token_batch = tokenizer.batch_encode_plus(
    raw_texts,
    max_length=CFG['max_len'],
    pad_to_max_length=True,
    return_attention_masks=True,
    return_tensors='tf',
  )
  if flatten_output:
    return token_batch['input_ids'], token_batch['attention_mask']
  else:
    return token_batch


class LangEncoder(tf.keras.Model):
  def __init__(self):
    super(LangEncoder, self).__init__()
    self._name = 'code_embed'
    self.lm = get_transformer(LM=False)


  def call(self, tokens, training=True):
    """Embed all tokens and return the average token embedding.
    """
    features = self.lm(tokens, training=training)[0]
    features = tf.math.reduce_mean(features, axis=1)
    return features


class LangDecoder(tf.keras.Model):
  def __init__(self):
    super(LangDecoder, self).__init__()
    self._name = 'code_decode'
    self.lm = get_transformer(LM=True)

  def call(self, features, training=True):
    """Return tokens based on a batch of embeddings.

    Features are of shape [batch_size, hidden_size]. Before predicting, this
    function tiles the features to equal max_len.
    """
    features = tf.expand_dims(features, axis=1)
    features = tf.tile(features, [1, CFG['max_len'], 1])
    logits = self.lm(None, inputs_embeds=features, training=training)[0]
    return logits


def sequence_reconstruction_loss(logits, code_tokens):
  """Loss an entire sequence predicted in one step. Evaluates each token
  individually.
  """
  vocab_size = logits.shape[2]
  input_ids = code_tokens['input_ids']
  labels = tf.one_hot(input_ids, depth=vocab_size)
  loss = tf.keras.losses.categorical_crossentropy(
    labels,
    logits,
    from_logits=True,
    label_smoothing=CFG['label_smoothing']
  )
  loss = tf.math.reduce_sum(loss)
  return loss


def sequence_reconstruction_accuracy(logits, code_tokens):
  vocab_size = logits.shape[2]
  input_ids = code_tokens['input_ids']
  labels = tf.one_hot(input_ids, depth=vocab_size)
  loss = tf.keras.losses.categorical_accuracy(
    labels,
    logits,
    from_logits=True,
    label_smoothing=CFG['label_smoothing']
  )
  loss = tf.math.reduce_sum(loss)
  return loss


if __name__ == "__main__":
  # test language embedding and decoding
  code_embed = LangEncoder()
  code_decode = LangDecoder()
  code_strings = [
    "def f(a): return a",
    "def f(b): return b"
  ]
  code_tokens = batch_encode(code_strings)
  embeddings = code_embed(code_tokens)
  logits = code_decode(embeddings)
  recon_loss = sequence_reconstruction_loss(logits, code_tokens)
