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
  return cfg


def set_config(cfg):
  global CFG
  CFG = cfg


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
    return CFG

if __name__ == "__main__":
  CFG = get_config()
