import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

    # parser.add_argument('filename')           # positional argument
    parser.add_argument('-s', '--stend')      # option that takes a value
    parser.add_argument('-c', '--cuda_devices')      # option that takes a value
    parser.add_argument('-m', '--mode', default='train')      # option that takes a value

    args = parser.parse_args()

    stend = args.stend
    mode = args.mode
    cuda_devices = args.cuda_devices.split('/')

    print(f'Starting at stend {stend.upper()}. Using {cuda_devices} cuda devices. Mode: {mode.upper()}')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cuda_devices)

