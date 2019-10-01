
import argparse
import os
import socket
import subprocess as sp
import sys
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models_yaml', help='file with models info, yaml format', type=str)
    parser.add_argument('--suppress_upload', help='don\'t upload any files', action='store_true')
    args = parser.parse_args()

    with open(args.models_yaml) as f:
        models = yaml.load(f, Loader=yaml.SafeLoader)

    placeholder = 'placeholder.csv'
    gs_base_path = 'gs://new_tpu_storage'

    for model in models:
        model_dir = gs_base_path + '/predicts/' + model["version"] + '-' + model["dataset"] + '-' \
                  + str(model['resolution'])
        res = sp.run(['gsutil', 'stat', model_dir + '/' + placeholder])

        if res.returncode == 0:
            print('\n' + model_dir + ' exists')
            continue


        print('\n' + model_dir + ' does not exist, starting inference')

        sp.run(['touch', placeholder], check=True)
        sp.run(['gsutil', 'cp', placeholder, model_dir + '/'], check=True)

        resolutions = [model['resolution']]

        for img_res in resolutions:
            model_name = model["version"] + '-' + model["dataset"] + '-' + str(img_res)
            gs_model_dir = gs_base_path + '/final/' + model_name

            dst_model_dir = 'models/' + model_name

            if not os.path.exists(dst_model_dir):
                os.makedirs(dst_model_dir)
                sp.run(['gsutil', '-m', 'cp', '-r', gs_model_dir + '/*', dst_model_dir],
                       check=True)

            os.makedirs('predictions', exist_ok=True)
            dst_predict = 'predictions/' + model_name + '.pkl'

            CHECK = False
            if socket.gethostname() in ['cppg', 'devbox']:
                sp.run(['./docker_run.sh', 'python', 'inference.py', dst_model_dir,
                        '--destination', dst_predict], check=CHECK)
            else:
                sp.run([sys.executable, 'inference.py', dst_model_dir, '--destination', dst_predict],
                       check=CHECK)

            if not os.path.exists(dst_predict):
                dst_predict = '../' + dst_predict

            if not args.suppress_upload:
                print('prediction finished, uploading the model')
                sp.run(['gsutil', '-m', 'cp', '-r', dst_predict, gs_base_path + '/predicts/'])
