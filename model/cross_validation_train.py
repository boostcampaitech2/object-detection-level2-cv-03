import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='train model file path')
    return args


def main(model):
    for cv_ver in range(1,6):
        subprocess.call['nohup','python3','tools/train.py',f'boostcamp/{model}.py','--cfg-options',f'{cv_ver}','&']
    
    ## ensemble 코드 추가

if __name__=='__main__':
    args = parse_args()
    main(args.model)