import sys
import yaml
import argparse
from pykeen.pipeline import pipeline

if __name__ == '__main__':

    # setting arguments
    parser = argparse.ArgumentParser(
        prog="pykeen_pipeline",
        usage="python script_pykeen_pipeline.py -i <parameter file> -o <dir_save>", 
        description="script for knowledge graph embedding with pykeen", 
        epilog="end",
        add_help=True, 
    )

    parser.add_argument("-i", "--f_param",    type=str, help="f_param")
    parser.add_argument("-o", "--dir_output", type=str, help="dir_output")

    args = parser.parse_args()

    # read parameters from file
    with open(args.f_param, 'r') as fin:
        dict_param = yaml.safe_load(fin)

    print(dict_param)

    # calculate knowledge graph embedding
    result = pipeline(**dict_param)

    # save result
    result.save_to_directory(args.dir_output)

    