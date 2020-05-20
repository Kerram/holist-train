import argparse
import os
import string
import random
import sys
import GPUtil
import shutil

parser = argparse.ArgumentParser(description='Runs fine tuning')
parser.add_argument('--docker-image', type=str, required=False)
parser.add_argument('--model-name', type=str, required=False)
parser.add_argument('--description', type=str, required=True, help="String, words separated by space")
parser.add_argument('--max-len-name', type=int, required=False, default=20)
parser.add_argument('--pack-to', type=str, required=False, default='/home/zpp/models_after_fine_tuning/')
args = parser.parse_args()

current_folder = os.path.dirname(os.path.realpath(__file__))

EXIT_CODE = 1

def query_yes_no(question, default="yes"):
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def determine(args):
    def get_or_else(val, default):
        if val is not None:
            return val
        return default

    def generate_random(n):
        return ''.join(random.choice(string.ascii_lowercase) for _ in range(n))

    description_fuscated = ''.join(ch for ch in args.description if ch.isalnum()).lower()
    base_name = int((args.max_len_name * 3) / 4)
    rest = int(args.max_len_name - base_name)
    custom_name = (''.join(description_fuscated.split(' '))[0:base_name]) + "-" + generate_random(rest)
    model_name = get_or_else(args.model_name, custom_name)
    docker_image = get_or_else(args.docker_image, custom_name)
    description = args.description

    available_gpus = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.2, maxMemory=0.2, includeNan=False)
    if len(available_gpus) == 0:
        print("Currently there is no gpu available, printing load")
        GPUtil.showUtilization()
        exit(EXIT_CODE)
    id_free_gpu = available_gpus[0]

    print('Determined starting arguments:')
    print(f'description = [{description}]')
    print(f'model_name = [{model_name}]')
    print(f'docker_image = [{docker_image}]')
    print(f'id_free_gpu = [{id_free_gpu}]')
    GPUtil.showUtilization()

    if not query_yes_no('Do you accept these?'):
        print("Right, start again [might want to --help then]")
        exit(EXIT_CODE)

    return (description_fuscated + '.' + model_name), model_name, docker_image, id_free_gpu
description, model_name, docker_image, id_free_gpu = determine(args)

os.system(f'mkdir -p /home/zpp/models_after_fine_tuning/{model_name}/best_eval')
os.system(f'docker build -f Dockerfile -t {docker_image} .')
print("--- Made dockerfile")
print("--- Attempting to zip folder and run docker container afterwards")
shutil.make_archive(args.pack_to + f'/{model_name}/' + description, 'zip', current_folder)
print("--- Zipped")
print("--- Running new wemux session")
os.system(f'wemux new-s -d -s {docker_image}')
run_command = f'docker run --gpus \'device={id_free_gpu}\' -v ' \
              '/home/zpp/deephol-data:/home/data -v ' \
              f'/home/zpp/models_after_fine_tuning/{model_name}:/home/model {docker_image}'
os.system(f'wemux send-keys -t {docker_image}:0 \'{run_command}\' Enter')
print(f"--- Wemux run on {docker_image} with command = [{run_command}]")
