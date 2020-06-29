import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import code
import yaml

import tensorflow as tf
import streamlit as st
import plotly.express as px

from cfg import get_config, set_config, validate_cfg; local_cfg = get_config(save=False)
from upload import gcs_download_blob_as_string

model_name = st.sidebar.text_input("Model name", value="cloud_prod_64")


def write_print(string):
  st.write(string)
  print(string)


def configure():
  config_path = f"logs/{model_name}/config.yaml"
  config_str = gcs_download_blob_as_string(config_path)
  model_cfg = yaml.safe_load(config_str)
  validate_cfg(model_cfg)
  model_cfg['batch_size'] = 1
  set_config(model_cfg)
  return model_cfg

model_cfg = configure()

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def obtain_model(model_name):  
  from models import get_model
  st.markdown("## Model Config")
  st.write(model_cfg)
  write_print("Building model architecture")
  model = get_model()

  def load_weights():
    cache_path = os.path.join(f"model_cache/{model_name}/best")
    if os.path.exists(os.path.dirname(cache_path)):
      write_print("Loading local weights")
      model.load_weights(cache_path)
    else:
      write_print("Loading remote weights")
      cloud_weights_path = f"{local_cfg['path_prefix']}checkpoints/{model_name}/best"
      model.load_weights(cloud_weights_path)
    # cache locally
    if not os.path.exists(os.path.dirname(cache_path)):
      write_print("Cacheing model weights locally")
      model.save_weights(cache_path)
  load_weights()
  return model


def obtain_tokenizer():
  from language import tokenizer
  return tokenizer


model = obtain_model(model_name)
tokenizer = obtain_tokenizer()

input_text = st.text_area("Enter some code")
if len(input_text) > 0:
  tokens = tokenizer.encode_plus(
    input_text,
    return_tensors='tf',
    return_attention_mask=True,
    max_length=48,
    pad_to_max_length=True
    )
  st.write(tokens)
  input_ids = tokens['input_ids']
  st.write("### Token Indices")
  st.write(input_ids)
  # st.write(attn)
  st.write("### Token strings")
  st.write(", ".join([tokenizer.decode(int(token)) for token in input_ids[0]]))

  # evaluate and populate
  result = model(tokens)
  Z = result["metadata"]["Z"][0]
  img = result["metadata"]["imgs"][0]
  Z_viz = tf.reshape(Z, [model_cfg['target_token_dim'], -1])
  Z_fig = px.imshow(Z_viz)
  st.write(Z_fig)
  img_fig = px.imshow(img)
  print(tf.math.reduce_max(img))
  print(tf.math.reduce_min(img))
  st.write(img_fig)