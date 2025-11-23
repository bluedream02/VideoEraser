import subprocess
import yaml

# YAML configuration file path
config_file = "configs/sample-vis.yaml"

# Loop to generate seeds from 0 to 100
for seed in range(101):  # From 0 to 100
    print(seed)
    # Read YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Modify seed value
    config['seed'] = seed

    # Write modified configuration back to file
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    # Execute command line to run python script
    command = f"python pipelines/sample.py --config {config_file}"
    subprocess.run(command, shell=True)
    print("end")
