from __future__ import annotations
from pathlib import Path
import yaml, os

def user_config_path() -> Path:
    return Path.home()/'.trident'/'config.yaml'

def default_data_root() -> Path:
    return Path.home()/'Documents'/'TRIDENT_Data'

def read_user_config() -> dict:
    p = user_config_path()
    if p.exists():
        try:
            return yaml.safe_load(p.read_text()) or {}
        except Exception:
            return {}
    return {}

def write_user_config(cfg: dict):
    p=user_config_path(); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))

def get_data_root() -> Path:
    env = os.environ.get('TRIDENT_DATA_ROOT')
    if env:
        return Path(env).expanduser()
    cfg=read_user_config()
    return Path(cfg.get('data_root', default_data_root())).expanduser()

def set_data_root(path: str|Path) -> Path:
    root=Path(path).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    cfg=read_user_config(); cfg['data_root']=str(root); write_user_config(cfg)
    ensure_tree(root)
    return root

def ensure_tree(root: Path):
    for sub in ['cache','data/raw','data/reference/osu','data/processed','reports/trident','figures','exports','logs']:
        (root/sub).mkdir(parents=True, exist_ok=True)
