
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

    with open('ready_models.txt') as f:
        ready_models = set(s.rstrip() for s in f)

    for model in models:
        img_res = str(model['resolution'])
        model_name = model['version'] + '-' + model['dataset'] + '-' + img_res
        gs_placeholder = gs_base_path + '/predicts/' + model_name + '_placeholder.csv'

        res = sp.run(['gsutil', 'stat', gs_placeholder])
        if res.returncode == 0:
            print('\n' + gs_placeholder + ' already exists')
            continue


        print('\n' + gs_placeholder + ' does not exist, starting inference')

        sp.run(['touch', placeholder], check=True)
        sp.run(['gsutil', 'cp', placeholder, gs_placeholder], check=True)

        gs_model_dir = gs_base_path + '/final/' + model_name

        local_model_dir = 'models/' + model_name
        dst_predict_path = 'predictions/' + model_name + '.pkl'
        gs_predict_path = gs_base_path + '/predicts/' + model_name + '.pkl'


        # check whether we have this model locally
        if model_name in ready_models:
            print(model_name + ' already exists locally')
            continue

        # check whether we have this model in the cloud
        res = sp.run(['gsutil', 'stat', gs_predict_path])
        if res.returncode == 0:
            print(gs_predict_path + ' already exists in the cloud')
            continue


        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir)
            sp.run(['gsutil', '-m', 'cp', '-r', gs_model_dir + '/*', local_model_dir],
                   check=True)

        os.makedirs('predictions', exist_ok=True)


        CHECK = False

        if socket.gethostname() in ['cppg', 'devbox']:
            res = sp.run(['./docker_run.sh', 'python', 'inference.py', local_model_dir,
                          '--destination', dst_predict_path], check=CHECK)
        else:
            res = sp.run([sys.executable, 'inference.py', local_model_dir, '--destination',
                          dst_predict_path], check=CHECK)

        if res.returncode != 0:
            sp.run(['gsutil', 'rm', gs_placeholder])
            continue

        if not os.path.exists(dst_predict_path):
            dst_predict_path = '../' + dst_predict_path

        if not args.suppress_upload:
            print('prediction finished, uploading the model')
            sp.run(['gsutil', 'cp', dst_predict_path, gs_predict_path])
