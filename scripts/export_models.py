
import argparse
import subprocess
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models_yaml', help='file with models info, yaml format', type=str)
    args = parser.parse_args()

    with open(args.models_yaml) as f:
        models = yaml.load(f, Loader=yaml.SafeLoader)

    for res in [640, 800, 1024]:
        for model in models:
            subprocess.run(['./export_saved_model.sh',
                            model['dataset'],
                            model['version'],
                            str(model['step']),
                            str(res)])
