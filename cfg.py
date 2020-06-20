import os
import yaml

file_dir = os.path.dirname(os.path.abspath(__file__))


def read_config():
  cfg = yaml.safe_load(open(
    os.path.join(file_dir, 'config.yaml'), 'r')
  )
  return cfg


def validate_cfg(cfg):
  assert cfg['full_model'] in cfg['full_models'], f"Model type {cfg['full_model']} not understood"
  if cfg['use_perceptual_loss']:
    assert cfg['full_model'] in ['vision', 'gestalt'], "Can't use perceptual loss without vision components"


def populate_cfg(cfg):
  cfg['ideal_vision_model_size'] = cfg['vision_model_size']
  if cfg['use_gcs']:
    gcs_credentials_path = os.path.join(file_dir, "gestalt-graph-59b01bb414f3.json")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcs_credentials_path
    cfg['path_prefix'] = cfg['gcs_prefix']
  else:
    cfg['path_prefix'] = cfg['local_prefix']
  # make some changes to parameters if using gestalt model
  if cfg['full_model'] == 'gestalt':
    target_token_dim = ((cfg['vision_model_size'] // cfg['max_len']) // 8) * 8
    new_vision_model_size = target_token_dim * cfg['max_len']
    cfg['target_token_dim'] = target_token_dim
    cfg['vision_model_size'] = new_vision_model_size
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


def get_config():
  global CFG
  try:
    CFG
    return CFG
  except NameError:
    cfg = read_config()
    validate_cfg(cfg)
    cfg = populate_cfg(cfg)
    set_config(cfg)
    save_config(cfg)
    return CFG


if __name__ == "__main__":
  CFG = get_config()
