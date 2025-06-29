import yaml

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Loads the config from the config file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config