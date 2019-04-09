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

from icecube import dataio, icetray, WaveCalibrator
from icecube import dataclasses, paraboloid, simclasses, recclasses, spline_reco
from icecube.trigger_sim.modules.time_shifter import I3TimeShifter
from I3Tray import *
from scipy.stats import moment, skew, kurtosis
import numpy as np
import math
import tables
import argparse
import os, sys
from configparser import ConfigParser
from lib.reco_quantities import *
from lib.functions_create_dataset import read_variables,cuts, get_stream, get_most_E_muon_info, median, get_t0
#import lib.transformations
import cPickle as pickle
import random
import lib.ic_grid as fu
import time
import logging
from icecube.phys_services.which_split import which_split
import time


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
        "--version",
        action="version", version='%(prog)s - Version 1.0')
    args = parser.parse_args()
    return args


args = parseArguments().__dict__

dataset_configparser = ConfigParser()
try:
    dataset_configparser.read(args['dataset_config'])
    print "Config is found {}".format(dataset_configparser)
except Exception:
    raise Exception('Config File is missing!!!!')

# configer the logger
logger = logging.getLogger('failed_frames')
logger_path = str(dataset_configparser.get('Basics', 'logger_path'))
if not os.path.exists(logger_path):
    os.makedirs(logger_path)
hdlr = logging.FileHandler(os.path.join(logger_path, 'failed_frames.log'))
formatter = logging.Formatter('%(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

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
scale_class = dict()
if 'Scale_Class' in dataset_configparser.keys():
    for key in dataset_configparser['Scale_Class'].keys():
        scale_class[int(key)] = int(dataset_configparser['Scale_Class'][key])
print('Scale classes like {}'.format(scale_class))
if len(scale_class.keys()) > 0:
    max_scale = np.max([scale_class[key] for key in scale_class])
else:
    max_scale = 1

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
    quantiles_pulses = np.linspace(0, 1 - float(pulses_input['step_size_pulses']),
                                  (1 / float(pulses_input['step_size_pulses'])))
    for q in quantiles_pulses:
        q = np.round(q, 3)
        inputs.append(('{}_{}_pct_charge_quantile'.format('pulse', str(q).strip().replace('.', '_')),
                       'pulses_quantiles(charges, times, {})'.format(q))) # pulses that are passed as argument needs to be defined



# This is the dictionary used to store the input data
events = dict()
events['reco_vals'] = []
events['pulses'] = []
events['waveforms'] = []
events['pulses_timeseries'] = []
events['t0'] = []


def save_to_array(phy_frame):
    """Save the waveforms pulses and reco vals to lists.

    Args:
        phy_frame, and I3 Physics Frame
    Returns:
        True (IceTray standard)
    """
    reco_arr = []
    if not z['ignore']:
        wf = None
    pulses = None
    if phy_frame is None:
        print('Physics Frame is None')
        return False
    for el in settings:
        if not z['ignore']:
            print z['ignore']
            if el[1] == '["CalibratedWaveforms"]':
                try:
                    wf = phy_frame["CalibratedWaveforms"]
                except Exception as inst:
                    print('uuupus {}'.format(el[1]))
                    print inst
                    return False
        elif el[1] == pulsemap_key:
            try:
                pulses = phy_frame[pulsemap_key].apply(phy_frame)
            except Exception as inst:
                print('Failed to add pulses {}'.format(el[1]))
                print inst
                print('Skip')
                return False
        elif el[0] == 'variable':
            try:
                reco_arr.append(eval('phy_frame{}'.format(el[1])))
            except Exception as inst:
                print('Failed to append Reco Vals {}'.format(el[1]))
                print inst
                print('Skip')
                return False
        elif el[0] == 'function':
            try:
                reco_arr.append(
                    eval(el[1].replace('_icframe_', 'phy_frame, geometry_file')))
            except Exception as inst:
                print('Failed to evaluate function {}'.format(el[1]))
                print(inst)
                print('Skip')
                return False

        # Removed part to append waveforms as it is depreciated
    if pulses is not None:
        tstr = 'Append Values for run_id {}, event_id {}'
        eheader = phy_frame['I3EventHeader']
        print(tstr.format(eheader.run_id, eheader.event_id))
        events['t0'].append(get_t0(phy_frame))
        events['pulses'].append(pulses)
        events['reco_vals'].append(reco_arr)
    else:
        print('No pulses in Frame...Skip')
        return False
    return


def event_picker(phy_frame):
    try:
        e_type = classify(phy_frame, geometry_file)
    except Exception as inst:
        print('The following event could not be classified')
        print(phy_frame['I3EventHeader'])
        print('First particle {}'.format(phy_frame['I3MCTree'][0].pdg_encoding))
	print(inst)
        return False
    rand = np.random.choice(range(1, max_scale+1))
    if e_type not in scale_class.keys():
        scaling = max_scale
    else:
        scaling = scale_class[e_type]
    if scaling >= rand:
        return True
    else:
        return False


def produce_data_dict(i3_file, num_events):
    """IceTray script that wraps around an i3file and fills the events dict
       that is initialized outside the function

    Args:
        i3_file, and IceCube I3File
    Returns:
        True (IceTray standard)
    """

    tray = I3Tray()
    tray.AddModule("I3Reader", "source",
                   FilenameList=[geometry_file,
                                 i3_file])

    if False:  # only needed if waveforms are used
        tray.AddModule(get_stream, "get_stream",
                       Streams=[icetray.I3Frame.Physics])


        tray.AddModule(event_picker, "event_picker",
                       Streams=[icetray.I3Frame.Physics])
        tray.AddModule("Delete",
                       "old_keys_cleanup",
                       keys=['CalibratedWaveformRange'])
        tray.AddModule("I3WaveCalibrator", "sedan",
                       Launches=waveform_key,
                       Waveforms="CalibratedWaveforms",
                       Errata="BorkedOMs",
                       ATWDSaturationMargin=123,
                       FADCSaturationMargin=0,)
    tray.AddModule(cuts, 'cuts',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(get_most_E_muon_info, 'get_most_E_muon_info',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(save_to_array, 'save',
                   Streams=[icetray.I3Frame.Physics])
    if num_events == -1:
        tray.Execute()
    else:
        tray.Execute(num_events)
    tray.Finish()
    return


def average(x, y):
    if len(y) == 0 or np.sum(y) == 0:
        return 0
    else:
        return np.average(x, weights=y)


if __name__ == "__main__":

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
        if len(args['filelist']) > 1:
            filelist = []
            for i in xrange(len(args['filelist'])):
                a = pickle.load(open(args['filelist'][i], 'r'))
                filelist.append(a)
            outfile = args['filelist'][0].replace('.pickle', '.h5')

        elif args['filelist'] is not None:
            filelist = pickle.load(open(args['filelist'][0], 'r'))
            outfile = args['filelist'][0].replace('.pickle', '.h5')
    elif args['files'] is not None:
        filelist = [args['files']]
        if filelist[0][0].split('/')[-1][-3:] == "zst":
            outfile = os.path.join(outfolder,filelist[0][0].split('/')[-1].replace('.i3.zst', '.h5'))
        if filelist[0][0].split('/')[-1][-3:] == "b2z":
            outfile = os.path.join(outfolder,filelist[0][0].split('/')[-1].replace('.i3.bz2', '.h5'))
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
        skipped_frames = 0
        statusInFilelist = 0
        event_files = []
        starttime = time.time()
        print len(filelist[0])
        print filelist
        while statusInFilelist < len(filelist[0]):
            timestamp = time.time()
            events['reco_vals'] = []
            if not z['ignore']:
                events['waveforms'] = []
            events['pulses'] = []
            counterSim = 0
            while counterSim < len(filelist):
                print('Attempt to read {}'.format(filelist[counterSim][statusInFilelist]))
                print('File to read Type {}'.format(type(filelist[counterSim][statusInFilelist])))
                print "Number of Events {}".format(args['max_num_events'])
                t3 = time.time()
                produce_data_dict(str(filelist[counterSim][statusInFilelist]),
                                  args['max_num_events'])
                counterSim = counterSim + 1
            print('--- Run {} --- Countersim is {} --'.format(statusInFilelist,
                                                              counterSim))
            statusInFilelist += 1
            # shuffeling of the files
            num_events = len(events['reco_vals'])
            print('The I3 File has {} events'.format(num_events))
            shuff = np.random.choice(num_events, num_events, replace=False)
            for i in shuff:
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
                    for dom in DOM_list:
                        gpos = grid[dom]
                        if dom in final_dict:
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

        print("\n -----------------------------")
        print('###### Run Summary ###########')
        print('Processed: {} Frames \n Skipped {}'.format(TotalEventCounter,
                                                          skipped_frames))
        print("-----------------------------\n")
        print("Finishing...")
        h5file.root._v_attrs.len = TotalEventCounter
    h5file.close()
