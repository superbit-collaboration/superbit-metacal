from pathlib import Path

from superbit_lensing import utils

'''
This script is for printing the global SuperBIT OBA config on the QCC during
flight operations. It takes no inputs and assumes that the OBA code has been
installed
'''

def main():

    config_file = Path(utils.MODULE_DIR) / 'oba/configs/oba_global_config.yaml'

    if not config_file.is_file():
        print('Failed: {config_file} not found')
        return 1

    config = utils.read_yaml(config_file)

    for outer_key in config:
        print(f'{outer_key}:')
        if isinstance(config[outer_key], dict):
            for inner_key, val in config[outer_key].items():
                print(f'  {inner_key}: {val}')
        elif isinstance(config[outer_key], list):
            for item in config[outer_key]:
                print(f'  -{item}')
        else:
            print(config[outer_key])

    return 0

if __name__ == '__main__':
    rc = main()

