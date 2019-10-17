# coding: utf-8

import argparse
import os, sys
from configparser import ConfigParser
from lib.functions_create_dataset import read_variables
import numpy as np
import importlib
print('pythonpath {}'.format(os.environ['pythonpath']))
sys.path.insert(0, os.environ['pythonpath'])

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
        "--outfile",
        help="name of the outfile",
        type=str, default='classification.npy')
    parser.add_argument(
        "--gcd",
        help="gcd file to use",
        type=str)
    parser.add_argument(
        "--filelist",
        help="filelist to be processed",
        type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parseArguments()
    print('Outfile {}'.format(args.outfile))
    print('In Files {}'.format(args.files))
    dataset_configparser = ConfigParser()
    try:
        dataset_configparser.read(args.dataset_config)
        print "Config is found {}".format(args.dataset_config)
    except Exception as ex:
        raise Exception('Config File is missing or unreadable!!!!')
        print ex
    pulsemap_key = str(dataset_configparser.get('Basics', 'PulseSeriesMap'))
    dtype, settings = read_variables(dataset_configparser)
    dt_new = list(dtype.names)
    dt_new.extend(['skimming', 'cascade', 'through_going', 'starting', 'stopping'])
    dtype = np.dtype(zip(dt_new, ['<f8'] * len(dt_new)))
    settings.extend([('variable', u"[\"Deep_Learning_Classification\"][\'Skimming\']", [-np.inf, np.inf]),
                    ('variable', u"[\"Deep_Learning_Classification\"][\'Cascade\']", [-np.inf, np.inf]),
                    ('variable', u"[\"Deep_Learning_Classification\"][\'Through_Going_Track\']", [-np.inf, np.inf]),
                    ('variable', u"[\"Deep_Learning_Classification\"][\'Starting_Track\']", [-np.inf, np.inf]),
                    ('variable', u"[\"Deep_Learning_Classification\"][\'Stopping_Track\']", [-np.inf, np.inf])])
    i3tray_file = str(dataset_configparser.get('Basics', 'tray_script'))
    sys.path.append(os.path.dirname(i3tray_file))
    sys.path.append(os.getcwd()+"/"+os.path.dirname(i3tray_file))
    mname = os.path.splitext(os.path.basename(i3tray_file))[0]
    print(' import {}'.format(mname))
    process_i3 = importlib.import_module(mname) 
    res_dicts = []
    if args.files is not None:
        files = []
        for j in np.atleast_1d(args.files):
            if os.path.isdir(j):
                files.extend([os.path.join(j,i) for i in os.listdir(j) if '.i3' in i])
            else:
                files.append(j)
    else:
        with open(args.filelist, 'r') as f:
            files = sorted(f.read().splitlines())
    print files
    f = files[0]
    f_bpath = os.path.split(f)[0]
    if args.gcd is None:
        geo_files = sorted([os.path.join(f_bpath, i) for i in os.listdir(f_bpath) if i[-6:] ==  '.i3.gz'])
        if len(geo_files) > 0:
            use_geo = str(geo_files[0])
        else:
            use_geo = str(dataset_configparser.get('Basics', 'geometry_file'))
    else:
        use_geo = args.gcd

    res_dicts.append(process_i3.run(files, -1, settings, use_geo, pulsemap_key, do_classification=True)['reco_vals'])

    result =np.array([tuple(i) for i in np.concatenate(res_dicts)], dtype=dtype)
    print result
    np.save(args.outfile, result)
