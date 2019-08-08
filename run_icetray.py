import argparse
import os, sys
from configparser import ConfigParser
from lib.functions_create_dataset import *
import numpy as np
import importlib

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_config",
        help="main config file, user-specific",
        type=str, default='default.cfg')
    parser.add_argument(
        "--files",
        help="file list for processing",
        type=str, nargs='+')
    parser.add_argument(
        "--gcd",
        help="gcd file for the processing",
        type=str)
    parser.add_argument(
        "--outfile",
        help="name of the outfile",
        type=str, default='classification.npy')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parseArguments()

    dataset_configparser = ConfigParser()
    try:
        dataset_configparser.read(args.dataset_config)
        print "Config is found {}".format(dataset_configparser)
    except Exception as ex:
        raise Exception('Config File is missing or unreadable!!!!')
        print ex
    pulsemap_key = str(dataset_configparser.get('Basics', 'PulseSeriesMap'))
    dtype, settings = read_variables(dataset_configparser)
    i3tray_file = str(dataset_configparser.get('Basics', 'tray_script'))
    sys.path.append(os.path.dirname(i3tray_file))
    sys.path.append(os.getcwd()+"/"+os.path.dirname(i3tray_file))
    mname = os.path.splitext(os.path.basename(i3tray_file))[0]
    process_i3 = importlib.import_module(mname)
    print dtype
    print settings
    if args.gcd is None:
        args.gcd = str(dataset_configparser.get('Basics', 'geometry_file'))
    res_dicts = []
    for f in args.files:
        res_dicts.append(process_i3.run(str(f), -1 , settings, args.gcd, pulsemap_key, do_classification=True)) 
    
