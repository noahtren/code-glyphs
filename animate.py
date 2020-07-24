import os
import code
import yaml
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow as tf
import imageio
from tqdm import tqdm

from cfg import get_config, set_config, validate_cfg; local_cfg = get_config(save=False)
from upload import gcs_download_blob_as_string

model_name = 'gestalt_cnn_percept_longtest_w_1_contd_6'
debug = True

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

def get_weights(model):
  cache_path = os.path.join(f"model_cache/{model_name}")
  if os.path.exists(cache_path):
    print("Loading local weights")
    model.custom_load(cache_path)
  else:
    print("Loading remote weights")
    cloud_weights_path = f"{local_cfg['path_prefix']}checkpoints/{model_name}"
    model.custom_load(cloud_weights_path)
  # cache locally
  if not os.path.exists(cache_path):
    print("Cacheing model weights locally")
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


def obtain_model(model_name):  
  from models import get_model
  print("Building model architecture")
  model = get_model()
  dummy = get_tokens("def hello(): return 'Hello!'")
  model(dummy)
  get_weights(model)
  return model


def get_embed(input_text):
  tokens = get_tokens(input_text)
  input_ids = tokens['input_ids']
  if debug:
    print(tokens)
    print("### Token Indices")
    print(input_ids)
    print("### Token strings")
    print(", ".join([tokenizer.decode(int(token)) for token in input_ids[0]]))

  # evaluate and populate
  result = model(tokens)
  Z = result["metadata"]["Z"][0]
  return Z

def get_image(Z):
  img = model.generator(Z)
  img = tf.image.adjust_contrast(img, 2.5)
  img = tf.clip_by_value(img, 0, 1)
  return img


def interpolate(texts):
  Zs = [get_embed(text) for text in texts]
  import tensorflow_graphics.math.interpolation.bspline
  embeds = tf.stack(Zs)
  interp = tensorflow_graphics.math.interpolation.bspline.interpolate(
    tf.transpose(embeds),
    tf.range(0, embeds.shape[0], delta=0.05, dtype=tf.float32),
    3, True)
  imgs = [get_image(embed[tf.newaxis]) for embed in tqdm(interp)]
  os.makedirs('vis', exist_ok=True)
  writer = imageio.get_writer(f'vis/{model_name}.mp4', fps=30)
  for img in imgs:
    img = tf.cast(img[0] * 255, tf.uint8).numpy()
    writer.append_data(img)

  writer.close()


if __name__ == "__main__":
  do_interp = False
  model = obtain_model(model_name)
  if do_interp:
    texts = [
      "def hello_world():\n  print(\"Hello world\"",
      "def fib(n):\n  return 1 if n <= 1 else fib(n-1) + fib(n-2)",
      "def is_positive(n):\n  return True if n > 0 else False",
      "def sqrt(x):\n  return tf.math.sqrt(x)",
      "def swish(x):\n  return tf.sigmoid(x) * x",
      "def check_auth(name, passwd):\n  with open('users.txt', 'r') as f:\n    users = json.load(f)\n    return False if user not in users else users[user] == password",
      "def dumb_loop(n):\n  sum = 0\n  for i in range(n):\n    print(i)\n    sum += 1\n    print(sum)",
      "def predict(x):\n  x = tf.cast(x, tf.float32) / 255.\n  pred = model.predict(x[tf.newaxis])\n  pred = tf.nn.softmax(pred)\n  return labels[tf.argmax(pred).numpy()]",
      "def is_food(item):\n if item.edible return True else False",
      "def show(x)\n  import matplotlib.pyplot as plt\b  plt.imshow(x)\n  plt.show(x)",
      "def interact():\n  code.interact(local={**locals(), **globals()})",
      "@app.route('/hello')\ndef hello(request):\n  return HttpResponse('Hey there!')",
      "@app.route('/hello')\ndef hello(request):\n  return JsonResponse({'message': 'Hey there!', 'time': str(datetime.now())})",
      "def maybe_add_maybe_subtract(a, b):\n  if random.random() > 0.5:\n    return a + b\n  else:\n    return a - b"
    ]
    interpolate(texts)
  else:
    texts = {
      "hello": "def hello_world():\n  print(\"Hello, world!\")",
      "fib": "def fib(n):\n  return 1 if n <= 1 else fib(n-1) + fib(n-2)",
      "swish": "def swish(x):\n  return tf.sigmoid(x) * x",
      "relu": "def relu(x):\n  return tf.nn.relu(x)",
      "softmax": "def softmax(x):\n  return tf.nn.softmax(x)",
      "sigmoid": "def sigmoid(x):\n  return tf.nn.sigmoid(x)",
    }
    for name, text in texts.items():
      Z = get_embed(text)
      img = get_image(Z[tf.newaxis])
      img = tf.cast(img * 255, tf.uint8)[0]
      imageio.imwrite(f'vis/{name}.png', img)
