import yaml
from pathlib import Path
import os

PROJECT_ROOT =  "/mnt/storage/RSystemsBenchmarking/gitProject/Benchmark/"
CONFIG_PATH = Path(os.path.join(PROJECT_ROOT,"config","config.yaml"))


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, path=CONFIG_PATH):
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config_section(section, key, value, path=CONFIG_PATH):
    config = load_config(path)
    if section not in config:
        config[section] = {}
    config[section][key] = value
    save_config(config, path)


# Example Usage
if __name__ == "__main__":
    config = load_config()
    print("Current COCO Path:", config["paths"]["dataset"]["coco"]["base_image_path"])

    # Update endpoint or add new
    # update_config_section("endpoints", "NewEndpoint", "http://localhost:1234/NewEndpoint")

    # Add new path
    # update_config_section("paths", "new_path_key", "/some/new/path")

    # Verify update
    print("Updated config:", load_config())
