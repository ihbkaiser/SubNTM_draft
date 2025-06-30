import argparse
import yaml

def update_args(args, path):
    with open(path) as file:
        config = yaml.safe_load(file)
        if config:
            args = vars(args)
            args.update(config)
            args = argparse.Namespace(**args)


def read_text(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        return f.read().splitlines()