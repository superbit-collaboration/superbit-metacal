from argparse import ArgumentParser
from superbit_lensing import utils
from superbit_lensing.cookiecutter import CookieCutter

import ipdb

parser = ArgumentParser()

parser.add_argument('config_file', type=str,
                    help='The CookieCutter yaml config file to use')

def main(args):

    config_file = args.config_file
    config = utils.read_yaml(config_file)

    cookie = CookieCutter(config=config)
    ipdb.set_trace()
    cookie.go()

    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    rc = main(args)

    if rc == 0:
        print('test passed succesfully!')
    else:
        print(f'test failed w/ return code of {rc}')
