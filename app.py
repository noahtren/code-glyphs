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

model_name = st.sidebar.text_input("Model name", value="gestalt_cnn_percept_longtest_w_1_contd_6")
debug = st.checkbox('Debug')
# better validation accuracy: gestalt_cnn_percept_longtest_w_1_contd_6
# dist_percept_long_contd_4

def configure():
  config_path = f"logs/{model_name}/config.yaml"
  config_str = gcs_download_blob_as_string(config_path)
  model_cfg = yaml.safe_load(config_str)
  validate_cfg(model_cfg)
  model_cfg['batch_size'] = 1
  set_config(model_cfg)
  return model_cfg


model_cfg = configure()
from language import tokenizer


def write_print(string):
  st.write(string)
  print(string)


def get_weights(model):
  cache_path = os.path.join(f"model_cache/{model_name}")
  if os.path.exists(cache_path):
    write_print("Loading local weights")
    model.custom_load(cache_path)
  else:
    write_print("Loading remote weights")
    cloud_weights_path = f"{local_cfg['path_prefix']}checkpoints/{model_name}"
    model.custom_load(cloud_weights_path)
  # cache locally
  if not os.path.exists(cache_path):
    write_print("Cacheing model weights locally")
    model.custom_save(cache_path)


def get_tokens(input_text):
  tokens = tokenizer.encode_plus(
    input_text,
    return_tensors='tf',
    return_attention_mask=True,
    max_length=model_cfg['max_len'],
    pad_to_max_length=True
  )
  return tokens


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def obtain_model(model_name):  
  from models import get_model
  st.markdown("## Model Config")
  st.write(model_cfg)
  write_print("Building model architecture")
  model = get_model()
  dummy = get_tokens("def hello(): return 'Hello!'")
  model(dummy)
  get_weights(model)
  return model


model = obtain_model(model_name)
input_text = st.text_area("Enter some code")

def do_prediction():
  tokens = get_tokens(input_text)
  input_ids = tokens['input_ids']
  if debug:
    st.write(tokens)
    st.write("### Token Indices")
    st.write(input_ids)
    st.write("### Token strings")
    st.write(", ".join([tokenizer.decode(int(token)) for token in input_ids[0]]))

  # evaluate and populate
  result = model(tokens)
  Z = result["metadata"]["Z"][0]
  img = result["metadata"]["imgs"][0]
  img = tf.image.adjust_contrast(img, 2.0)
  img = tf.clip_by_value(img, 0, 1)
  Z_viz = tf.reshape(Z, [model_cfg['target_down_token_dim'], -1])
  if debug:
    Z_fig = px.imshow(Z_viz)
    st.write(Z_fig)
  img_fig = px.imshow(img)
  st.write(img_fig)

if len(input_text) > 0:
  do_prediction()