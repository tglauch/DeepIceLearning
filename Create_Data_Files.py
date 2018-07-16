#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v2/icetray-start
#METAPROJECT /data/user/mkronmueller/Software/combo/build
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
from I3Tray import *
from scipy.stats import moment, skew, kurtosis
import numpy as np
import math
import tables
import argparse
import os, sys
from configparser import ConfigParser
from reco_quantities import *
import cPickle as pickle
import random
import functions_Create_Data_Files as fu
import time
import logging

def replace_with_var(x):
    y = x.replace('c', 'charges').replace('t', 'times').replace('w', 'widths')
    return y


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
        type=int,  default=-1)    
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
dtype, settings = fu.read_variables(dataset_configparser)
settings.append(('variable', '["CalibratedWaveforms"]'))
settings.append(('variable', '["InIceDSTPulses"]'))

# Parse Input Features
x = dataset_configparser['Input_Charges']
y = dataset_configparser['Input_Times']
z = dataset_configparser['Input_Waveforms1']
inputs = []
for key in x.keys():
    inputs.append((key, replace_with_var(x[key])))
for key in y.keys():
    inputs.append((key, replace_with_var(y[key])))
for q in z['quantiles'].split(','):
    inputs.append(('{}_{}_pct_charge_quantile'.format(z['type'], q.strip().replace('.','_')),
                   'fu.wf_quantiles(waveform, {})[\'{}\']'.format(q, z['type'])))    

# This is the dictionary used to store the input data
events = dict()
events['reco_vals'] = []
events['waveforms'] = []
events['pulses'] = []


def save_to_array(phy_frame):
    """Save the waveforms pulses and reco vals to lists.

    Args:
        phy_frame, and I3 Physics Frame
    Returns:
        True (IceTray standard)
    """

    reco_arr = []
    wf = None
    pulses = None
    if phy_frame is None:
        print('Physics Frame is None')
        return False
    for el in settings:
        if el[1] == '["CalibratedWaveforms"]':
            try:
                wf = phy_frame["CalibratedWaveforms"]
            except Exception:
                print('uuupus {}'.format(el[1]))
                return False
        elif el[1] == '["InIceDSTPulses"]':
            try:
                pulses = phy_frame["InIceDSTPulses"].apply(phy_frame)
            except Exception:
                print('uuupus {}'.format(el[1]))
                return False
        elif el[0] == 'variable':
            try:
                reco_arr.append(eval('phy_frame{}'.format(el[1])))
            except Exception:
                print('uuupus {}'.format(el[1]))
                return False
        elif el[0] == 'function':
            try:
                reco_arr.append(
                    eval(el[1].replace('(x)', '(phy_frame, geometry_file)')))
            except Exception:
                print('uuupus {}'.format(el[1]))
                return False
        if (wf is not None) and (pulses is not None):
            print('Gut')
            events['waveforms'].append(wf)
            events['pulses'].append(pulses)
            events['reco_vals'].append(reco_arr)
    return


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
                   Filenamelist=[geometry_file,
                                 i3_file],)
    tray.AddModule("Delete",
                   "old_keys_cleanup",
                   keys=['CalibratedWaveformRange'])
    tray.AddModule(cuts, 'cuts', Streams=[icetray.I3Frame.Physics])
    tray.AddModule("I3WaveCalibrator", "sedan",
                   Launches="InIceRawData",
                   Waveforms="CalibratedWaveforms",
                   Errata="BorkedOMs",
                   ATWDSaturationMargin=123,
                   FADCSaturationMargin=0,)
    tray.AddModule(save_to_array, 'save', Streams=[icetray.I3Frame.Physics])
    if num_events==-1:
        tray.Execute()
    else:
        tray.Execute(num_events)
    tray.Finish()
    return


def cuts(phy_frame):
    """Performe a pre-selection of events according
       to the cuts defined in the config file

    Args:
        phy_frame, and IceCube I3File
    Returns:
        True (IceTray standard)
    """
    cuts = dataset_configparser['Cuts']
    particle_type = phy_frame['I3MCTree'][0].pdg_encoding
    RunID = phy_frame['I3EventHeader'].run_id
    EventID = phy_frame['I3EventHeader'].event_id
    # Checking for wierd event structures
    if testing_event(phy_frame, geometry_file) == -1:
        report = [particle_type, RunID, EventID, "EventTestingFailed"]
        logger.info(report)
        return False
    ParticelList = [12, 14, 16]
    if cuts['only_neutrino_as_primary_cut'] == "ON":
        if abs(phy_frame['MCPrimary'].pdg_encoding) not in ParticelList:
            report = [particle_type, RunID, EventID, "NeutrinoPrimaryCut"]
            logger.info(report)
            return False
    if cuts['max_energy_cut'] == "ON":
        energy_cutoff = cuts['max_energy_cutoff']
        if calc_depositedE(phy_frame) > energy_cutoff:
            report = [particle_type, RunID, EventID, "MaximalEnergyCut"]
            logger.info(report)
            return False
    if cuts['minimal_tau_energy'] == "ON":
        I3Tree = phy_frame['I3MCTree']
        primary_list = I3Tree.get_primaries()
        if len(primary_list) == 1:
            neutrino = I3Tree[0]
        else:
            for p in primary_list:
                pdg = p.pdg_encoding
                if abs(pdg) in ParticelList:
                    neutrino = p
        minimal_tau_energy = int(cuts['minimal_tau_energy'])
        if abs(neutrino.pdg_encoding) == 16:
            if calc_depositedE(phy_frame) < minimal_tau_energy:
                report = [particle_type, RunID, EventID, "MinimalTauEnergyCut"]
                flogger.info(report)
                return False
    if cuts['min_energy_cut'] == "ON":
        energy_cutoff = int(cuts['min_energy_cutoff'])
        if calc_depositedE(phy_frame) < energy_cutoff:
            report = [particle_type, RunID, EventID, "MinimalEnergyCut"]
            logger.info(report)
            return False
    if cuts['min_hit_DOMs_cut'] == "ON":
        if calc_hitDOMs(phy_frame) < cuts['min_hit_DOMs']:
            report = [particle_type, RunID, EventID, "HitDOMsCut"]
            logger.info(report)
            return False
    return True


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
        input_shape = [12, 11, 61]
        grid, DOM_list = fu.make_autoHexGrid(geo)

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
            # path of outfile could be changed to a new folder for a  better overview
            outfile = args['filelist'][0].replace('.pickle', '.h5')

        elif args['filelist'] is not None:
            filelist = pickle.load(open(args['filelist'], 'r'))
            outfile = args['filelist'].replace('.pickle', '.h5')
    elif args['files'] is not None:
        filelist = [args['files']]
        outfile = os.path.join(outfolder,filelist[0][0].split('/')[-1].replace('.i3.bz2', '.h5'))

    else:
        raise Exception('No input files given')

    if os.path.exists(outfile):
        os.remove(outfile)

    FILTERS = tables.Filters(complib='zlib', complevel=9)
    with tables.open_file(
        outfile, mode="w", title="Events for training the NN",
            filters=FILTERS) as h5file:
        input_features = []
        for inp in inputs:
            print 'Generate Input Feature {}'.format(inp[0])
            feature = h5file.create_earray(
                h5file.root, inp[0], tables.Float64Atom(),
                (0, input_shape[0], input_shape[1], input_shape[2], 1),
                title=inp[1])
            input_features.append(feature)
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
        while statusInFilelist < len(filelist[0]):
            timestamp = time.time()
            events['reco_vals'] = []
            events['waveforms'] = []
            events['pulses'] = []
            counterSim = 0
            while counterSim < len(filelist):
                try:
                    print('Attempt to read {}'.format(filelist[counterSim][statusInFilelist]))
                    produce_data_dict(filelist[counterSim][statusInFilelist], args['max_num_events'])
                    counterSim = counterSim + 1
                except Exception:
                    continue
            print('--- Run {} --- Countersim is {} --'.format(statusInFilelist,
                                                              counterSim))
            statusInFilelist += 1
            # shuffeling of the files
            num_events = len(events['reco_vals'])
            print('The I3 File has {} events'.format(num_events))
            shuff = np.random.choice(num_events, num_events, replace=False)
            for i in shuff:
                print i
                TotalEventCounter += 1
                reco_arr = events['reco_vals'][i]
                if not len(reco_arr) == len(dtype):
                    continue
                try:
                    reco_vals.append(np.array(reco_arr))
                except Exception:
                    continue

                pulses = events['pulses'][i]
                waveforms = events['waveforms'][i]
                final_dict = dict()
                for omkey in waveforms.keys():
                    if omkey in pulses.keys():
                        charges = np.array([p.charge for p in pulses[omkey][:]])
                        times = np.array([p.time for p in pulses[omkey][:]])
                        widths = np.array([p.width for p in pulses[omkey][:]])
                    else:
                        widths = np.array([0])
                        times = np.array([0])
                        charges = np.array([0])
                    waveform = waveforms[omkey]
                    final_dict[(omkey.string, omkey.om)] = \
                        [eval(inp[1]) for inp in inputs]
                for inp_c, inp in enumerate(inputs):
                    f_slice = np.zeros((1, input_shape[0], input_shape[1],
                                        input_shape[2], 1))
                    for dom in DOM_list:
                        gpos = grid[dom]
                        if dom in final_dict:
                            f_slice[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                final_dict[dom][inp_c]
                    input_features[inp_c] = f_slice

            print('Flush data to HDF File')
            for inp_feature in input_features:
                inp_feature.flush()
            reco_vals.flush()

        print("\n -----------------------------")
        print('###### Run Summary ###########')
        print('Processed: {} Frames \n Skipped {}'.format(TotalEventCounter,
                                                          skipped_frames))
        print("\n Frames with a I3MCTree Problem {}".format(TreeProblem))
        print("-----------------------------\n")
        print("Finishing...")
        h5file.root._v_attrs.len = TotalEventCounter
    h5file.close()
