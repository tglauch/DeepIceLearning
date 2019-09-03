# coding: utf-8

"""This file is part of DeepIceLearning
DeepIceLearning is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

'''
small differences when processing data for the diffuse dataset
'''

from icecube import dataio, dataclasses
from scipy.stats import moment, skew, kurtosis
import numpy as np
import tables
import argparse
import os, sys
from configparser import ConfigParser
from lib.functions_create_dataset import *
#import lib.transformations
import cPickle as pickle
import random
import lib.ic_grid as fu
import time
import importlib

def replace_with_var(x):
    """Replace the config parser input names with var names
       in the Code.

    Args:
        Transformation performed on charge time or pulse widths
    Returns:
        Correctly formatted transformation for the code
    """

    if ('(' in x) and (')' in x):
        y = x[x.index('('):x.index(')')]
        x = x.replace(y, y.replace('c', 'charges').
                      replace('t', 'times').replace('w', 'widths'))
    else:
        x = x.replace('c', 'charges').replace('t', 'times').\
            replace('w', 'widths')
    return x

# arguments given in the terminal


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_config",
        help="main config file, user-specific",
        type=str, default='default.cfg')
    parser.add_argument(
        "--files",
        help="files to be processed",
        type=str, nargs="+", required=False)
    parser.add_argument(
        "--max_num_events",
        help="The maximum number of frames to be processed",
        type=int, default=-1)
    parser.add_argument(
        "--filelist",
        help="Path to a filelist to be processed",
        type=str, nargs="+", required=False)
    parser.add_argument(
        "--memory_saving",
        help="If you want to save memory by only doing \
              len(filelist) events at the same time",
        action="store_true", default=False)
    parser.add_argument(
        "--version",
        action="version", version='%(prog)s - Version 1.0')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parseArguments().__dict__

    dataset_configparser = ConfigParser()
    try:
        dataset_configparser.read(args['dataset_config'])
        print "Config is found {}".format(dataset_configparser)
    except Exception as ex:
        raise Exception('Config File is missing or unreadable!!!!')
        print ex

    i3tray_file = dataset_configparser.get('Basics', 'tray_script')
    sys.path.append(os.path.dirname(i3tray_file))
    sys.path.append(os.getcwd()+"/"+os.path.dirname(i3tray_file))
    mname = os.path.splitext(os.path.basename(i3tray_file))[0]
    process_i3 = importlib.import_module(mname)

    # File paths
    geometry_file = str(dataset_configparser.get('Basics', 'geometry_file'))
    outfolder = str(dataset_configparser.get('Basics', 'out_folder'))
    pulsemap_key = str(dataset_configparser.get('Basics', 'PulseSeriesMap'))
    dtype, settings = read_variables(dataset_configparser)
    waveform_key = str(dataset_configparser.get('Basics', 'Waveforms'))
    if not dataset_configparser['Input_Waveforms1']['ignore']:
        settings.append(('variable', '["CalibratedWaveforms"]'))
    settings.append(('variable', '{}'.format(pulsemap_key)))
    print('Settings: {}'.format(settings))
    # Parse Input Features
    x = dataset_configparser['Input_Charges']
    y = dataset_configparser['Input_Times']
    z = dataset_configparser['Input_Waveforms1']
    pulses_input = dataset_configparser['Input_Pulses']

    inputs = []
    for key in x.keys():
        inputs.append((key, x[key]))
    for key in y.keys():
        inputs.append((key, y[key]))
    if not z['ignore']:
        quantiles = np.linspace(0, 1. - z['step_size'], (1. / z['step_size']))
        for q in quantiles:
            q = np.round(q, 2)
            inputs.append(('{}_{}_pct_charge_quantile'.format(z['type'], q.strip().replace('.', '_')),
                           'wf_quantiles(waveform, {})[\'{}\']'.format(q, z['type'])))
    print "Pulses Ignore: {}".format(pulses_input['ignore'])
    if pulses_input['ignore'] == "False":
        print pulses_input['quantiles']
        quantiles_pulses = np.array([float(i) for i in pulses_input['quantiles'].split(',')])
        for q in quantiles_pulses:
            q = np.round(q, 3)
            inputs.append(('{}_{}_pct_charge_quantile'.format('pulse', str(q).strip().replace('.', '_')),
                           'pulses_quantiles(charges, times, {})'.format(q)))

    # This is the dictionary used to store the input data
    events = dict()
    events['reco_vals'] = []
    events['pulses'] = []
    events['waveforms'] = []
    events['pulses_timeseries'] = []
    events['t0'] = []

    # Raw print arguments
    print("\n---------------------")
    print("You are running the script with arguments: ")
    for a in args.keys():
        print(str(a) + ": " + str(args[a]))
    print("---------------------\n")
    geo = dataio.I3File(geometry_file).pop_frame()['I3Geometry'].omgeo
    input_shape_par = dataset_configparser.get('Basics', 'input_shape')
    if input_shape_par != "auto":
        input_shape = eval(input_shape_par)
        grid, DOM_list = fu.make_grid_dict(input_shape, geo)
    else:
        input_shape = [10, 10, 60]
        grid, DOM_list = fu.make_stefans_grid(geo)

#    input_shape_DC = [5, 3, 60]
#    grid_DC, DOM_list_DC = fu.make_Deepcore_Grid(geo)

    # Create HDF5 File ##########
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    # default version, when using the submit script
    if args['filelist'] is not None:
        filelist = []
        for i in xrange(len(args['filelist'])):
            a = pickle.load(open(args['filelist'][i], 'r'))
            filelist.append(a)
        outfile = args['filelist'][0].replace('.pickle', '.h5')
    elif args['files'] is not None:
        filelist = [args['files']]
        if filelist[0][0].split('/')[-1][-3:] == "zst":
            outfile = os.path.join(outfolder,filelist[0][0].split('/')[-1].replace('.i3.zst', '.h5'))
        elif filelist[0][0].split('/')[-1][-3:] == "bz2":
            outfile = os.path.join(outfolder,filelist[0][0].split('/')[-1].replace('.i3.bz2', '.h5'))
        elif filelist[0][0].split('/')[-1][-3:] == ".i3":
            outfile = os.path.join(outfolder,filelist[0][0].split('/')[-1].replace('.i3', '.h5'))
        else:
            print "Take compreshion format of I3-File into account"

    else:
        raise Exception('No input files given')

    if os.path.exists(outfile):
        os.remove(outfile)

    FILTERS = tables.Filters(complib='zlib', complevel=9)
    print "OUT: {}".format(outfile)
    with tables.open_file(
        outfile, mode="w", title="Events for training the NN",
            filters=FILTERS) as h5file:
        input_features = []
        print "Inputs: {}".format(inputs)
        for inp in inputs:
            print 'Generate Input Feature {}'.format('IC_{}'.format(inp[0]))
            feature = h5file.create_earray(
                h5file.root, 'IC_{}'.format(inp[0]), tables.Float64Atom(),
                (0, input_shape[0], input_shape[1], input_shape[2], 1),
                title='IC_{}'.format(inp[1]))
            feature.flush()
#            print 'Generate Input Feature {}'.format('DC_{}'.format(inp[0]))        
            input_features.append(feature)
#            feature = h5file.create_earray(
#                h5file.root, 'DC_{}'.format(inp[0]), tables.Float64Atom(),
#                (0, input_shape_DC[0], input_shape_DC[1], input_shape_DC[2], 1),
#                title='DC_{}'.format(inp[1]))
#            feature.flush()
#            input_features.append(feature)
        reco_vals = tables.Table(h5file.root, 'reco_vals',
                                 description=dtype)
        h5file.root._v_attrs.shape = input_shape

        print('Created a new HDF File with the Settings:')
        print(h5file)
        print(h5file.root)

        np.save('grid.npy', grid)
        TotalEventCounter = 0
        statusInFilelist = 0
        starttime = time.time()

        # be careful here, if the filelists don't have the same length you will
        # get a dataset that is *not* completely shuffeled
        # if you want to have this guranteed, than either turn of the memory saving mode
        # or give filelists of the same length

        if not args['memory_saving']:
            filelist = np.concatenate(filelist)
            nloops = 1
        else:
            nloops = np.max([len(f) for f in filelist])


        # Generate Output
        print(filelist)
        while statusInFilelist < nloops:
            events['reco_vals'] = []
            if not z['ignore']:
                events['waveforms'] = []
            events['pulses'] = []
            counterSim = 0
            t0 = time.time()
            if not args['memory_saving']:
                for f in filelist:
                    print('Attempt to read {}'.format(f))
                    f_bpath = os.path.split(f)[0]
                    geo_files = sorted([os.path.join(f_bpath, i) for i in os.listdir(f_bpath) if i[-6:] ==  '.i3.gz'])
                    if len(geo_files) > 0:
                        use_geo = str(geo_files[0])
                    else:
                        use_geo = str(geometry_file)
                    print('Use Geo: {}'.format(use_geo))
                    t_dict = process_i3.run(str(f), args['max_num_events'], settings, use_geo, pulsemap_key)
                    for key in t_dict.keys():
                        events[key].extend(t_dict[key])                    
            else:
                while counterSim < len(filelist):
                    print('Attempt to read {}'.format(filelist[counterSim][statusInFilelist]))
                    print "Number of Events {}".format(args['max_num_events'])
                    f = str(filelist[counterSim][statusInFilelist])
                    f_bpath = os.path.split(f)[0]
                    geo_files = [os.path.join(f_bpath, i) for i in os.listdir(f_bpath) if i[-6:] ==  '.i3.gz']
                    if len(geo_files) > 0:
                        use_geo = str(geo_files[0])
                    else:
                        use_geo = str(geometry_file)
                    t_dict = process_i3.run(str(f), args['max_num_events'], settings, use_geo, pulsemap_key)
                    for key in t_dict.keys():
                        events[key].extend(t_dict[key])
                    counterSim = counterSim + 1
            print('--- Run {} --- Countersim is {} --'.format(statusInFilelist,
                                                              counterSim))
            statusInFilelist += 1
            # shuffeling of the files
            num_events = len(events['reco_vals'])
            t1 = time.time()
            dt = t1 - t0
            print('The I3 File(s) have {} events'.format(num_events))
            print('Processing took {:.1f} s'.format( dt))
            print('which equals {:.1f} ms per event'.format(1000.*dt/num_events))
            if 'shuffle' in dataset_configparser['Basics'].keys():
                if str(dataset_configparser.get('Basics', 'shuffle')) == 'True':
                    shuff = np.random.choice(num_events, num_events, replace=False)
                    print('Shuffle is ON')
                else:
                    shuff = range(num_events)
                    print('Shuffle is OFF')
            else:
                shuff = np.random.choice(num_events, num_events, replace=False)
                print('Shuffle is ON')

            for j, i in enumerate(shuff):
    
                if j%(np.max([1, int(1.*num_events/10.)])) == 0:
                    print('{:.1f} \%'.format(100.*j/num_events))
                TotalEventCounter += 1
                reco_arr = events['reco_vals'][i]
                if not len(reco_arr) == len(dtype):
                    print('Len of the reco array does not match the dtype')
                    continue
                try:
                    reco_vals.append(np.array(reco_arr))
                except Exception:
                    print('Could not append the reco vals')
                    print(reco_vals)
                    continue

                pulses = events['pulses'][i]
                if not z['ignore']:
                    waveforms = events['waveforms'][i]
                final_dict = dict()
                for omkey in pulses.keys():
                    charges = np.array([p.charge for p in pulses[omkey][:]])
                    times = np.array([p.time for p in pulses[omkey][:]]) - events['t0'][i]
                    widths = np.array([p.width for p in pulses[omkey][:]])
                    if not z['ignore']:
                        waveform = waveforms[omkey]
                    final_dict[(omkey.string, omkey.om)] = \
                        [eval(inp[1]) for inp in inputs]
                for inp_c, inp in enumerate(inputs):
                    f_slice = np.zeros((1, input_shape[0],
                                        input_shape[1],
                                        input_shape[2], 1))
                    for dom in final_dict:
                        if not dom in grid.keys():
                            continue
                        gpos = grid[dom]
                        f_slice[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                            final_dict[dom][inp_c]
                    input_features[inp_c].append(f_slice)

#                    f_slice = np.zeros((1, input_shape_DC[0],
#                                        input_shape_DC[1],
#                                        input_shape_DC[2], 1))
#                    for dom in DOM_list_DC:
#                        gpos = grid_DC[dom]
#                        if dom in final_dict:
#                            f_slice[0][gpos[0]][gpos[1]][gpos[2]][0] = \
#                                final_dict[dom][inp_c]
#                    input_features[2 * inp_c + 1].append(f_slice)

            print('Flush data to HDF File')
            for inp_feature in input_features:
                inp_feature.flush()
            reco_vals.flush()

        t2 = time.time()
        dt = t2 - t1
        print("\n -----------------------------")
        print('###### Run Summary ###########')
        print('Processed: {} Frames'.format(TotalEventCounter))
        print('Writing to HDF took {:.1f} s'.format(dt))
        print('which equals {:.1f} s per event'.format(1.*dt/num_events))
        print("-----------------------------\n")
        print("Finishing...")
        h5file.root._v_attrs.len = TotalEventCounter
    h5file.close()
