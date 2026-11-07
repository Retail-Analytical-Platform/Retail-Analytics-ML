from pathlib import Path
import yaml

config_dir = Path(__file__).parent.parent / 'config'
all_configs = {}

# Loop through all files ending in .yaml
for file_path in config_dir.glob('*.y*ml'):
    with open(file_path, 'r') as file:
        all_configs[file_path.stem] = yaml.safe_load(file)

print(all_configs)
