import os
import yaml

file_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def read_config():
  cfg = yaml.safe_load(open(
    os.path.join(file_dir, 'config.yaml'), 'r')
  )
  return cfg


def validate_cfg(cfg):
  assert cfg['full_model'] in cfg['full_models'], f"Model type {cfg['full_model']} not understood"
  if 'use_perceptual_loss' not in cfg:
    cfg['use_perceptual_loss'] = False
  if 'perceptual_loss_style' not in cfg:
    cfg['perceptual_loss_style'] = 'normal'
  if 'percept_mult' not in cfg:
    cfg['percept_mult'] = 1.0
  if 'vision_backbone' not in cfg:
    cfg['vision_backbone'] = 'BiT-M-R50x1'
  if 'generator_downsample_ratio' not in cfg:
    cfg['generator_downsample_ratio'] = 2
  if 'generator_model' not in cfg:
    cfg['generator_model'] = 'cnn'
  assert cfg['generator_model'] in cfg['generator_models']
  if cfg['use_perceptual_loss']:
    assert cfg['full_model'] in ['vision', 'gestalt'], "Can't use perceptual loss without vision components"


def populate_cfg(cfg):
  cfg['ideal_generator_model_size'] = cfg['generator_model_size']
  if cfg['use_gcs']:
    gcs_credentials_path = os.path.join(file_dir, "gestalt-graph-59b01bb414f3.json")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcs_credentials_path
    cfg['path_prefix'] = cfg['gcs_prefix']
  else:
    cfg['path_prefix'] = cfg['local_prefix']
  # make some changes to parameters if using gestalt model
  if cfg['full_model'] == 'gestalt':
    target_down_token_dim = ((cfg['generator_input_size'] // cfg['max_len']) // 8) * 8
    target_up_token_dim = ((cfg['vision_model_size'] // cfg['max_len']) // 8) * 8
    cfg['target_down_token_dim'] = target_down_token_dim
    cfg['target_up_token_dim'] = target_up_token_dim
    cfg['generator_input_size'] = target_down_token_dim * cfg['max_len']
  # TPU
  cfg['TPU'] = False
  if 'IS_TPU' in os.environ:
    if os.environ['IS_TPU'] == 'y':
      cfg['TPU'] = True
  return cfg


def set_config(cfg):
  global CFG
  CFG = cfg


def save_config(cfg):
  from upload import gcs_upload_blob_from_string
  log_dir = f"logs/{cfg['run_name']}"
  cfg_dir = os.path.join(log_dir, 'config.yaml')
  cfg_str = yaml.safe_dump(cfg)
  if cfg['use_gcs']:
    gcs_upload_blob_from_string(cfg_str, cfg_dir, print_str=True)
  else:
    with open(cfg_dir, 'w+') as f:
      f.write(cfg_str)


def get_config(save=True):
  global CFG
  try:
    CFG
    return CFG
  except NameError:
    cfg = read_config()
    validate_cfg(cfg)
    cfg = populate_cfg(cfg)
    set_config(cfg)
    if save:
      save_config(cfg)
    return CFG


if __name__ == "__main__":
  CFG = get_config()
