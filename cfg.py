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


def populate_cfg(cfg):
  if cfg['use_gcs']:
    gcs_credentials_path = os.path.join(file_dir, "gestalt-graph-59b01bb414f3.json")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcs_credentials_path
    cfg['path_prefix'] = cfg['gcs_prefix']
  else:
    cfg['path_prefix'] = cfg['local_prefix']
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
