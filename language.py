"""Encode/decode source code to/from a fixed-length representation.

Unfortunately not using CODEBert because it is only available in PyTorch.
Instead, using the generic distilBERT language model.
"""

import code

import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer, TFDistilBertForMaskedLM
from transformers import AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import TFRobertaModel, TFRobertaForMaskedLM
from transformers import load_pytorch_model_in_tf2_model

from cfg import get_config; CFG = get_config()

MODEL_NAMES = [
  'distilbert-base-cased',
  'huggingface/CodeBERTa-small-v1'
]


model_name = 'huggingface/CodeBERTa-small-v1'
if model_name == 'distilbert-base-cased':
  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
elif model_name == 'huggingface/CodeBERTa-small-v1':
  tokenizer = AutoTokenizer.from_pretrained('huggingface/CodeBERTa-small-v1')


def pt_to_tf(pt_model, tf_model_class):
  assert model_name == 'huggingface/CodeBERTa-small-v1'
  empty_tf_model = tf_model_class(pt_model.config)
  tf_model = load_pytorch_model_in_tf2_model(empty_tf_model, pt_model)
  return tf_model


def get_transformer(LM:bool):
  if LM:
    if model_name == 'distilbert-base-cased':
      model = TFDistilBertForMaskedLM.from_pretrained(
        'distilbert-base-cased'
      )
    elif model_name == 'huggingface/CodeBERTa-small-v1':
      model = AutoModelWithLMHead.from_pretrained(
        'huggingface/CodeBERTa-small-v1'
      )
      model = pt_to_tf(model, TFRobertaForMaskedLM)
  else:
    if model_name == 'distilbert-base-cased':
      model = TFDistilBertModel.from_pretrained(
        'distilbert-base-cased',
      )
    elif model_name == 'huggingface/CodeBERTa-small-v1':
      model = AutoModel.from_pretrained(
        'huggingface/CodeBERTa-small-v1'
      )
      model = pt_to_tf(model, TFRobertaModel)
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


def sequence_reconstruction_loss(code_tokens, logits):
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


def sequence_reconstruction_accuracy(code_tokens, logits):
  vocab_size = logits.shape[2]
  input_ids = code_tokens['input_ids']
  labels = tf.one_hot(input_ids, depth=vocab_size)
  loss = tf.keras.metrics.categorical_accuracy(
    labels,
    logits,
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
