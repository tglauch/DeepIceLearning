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

from icecube import dataio, icetray
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


##### used for later calculations

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

# File paths 
geometry_file = str(dataset_configparser.get('Basics', 'geometry_file'))
outfolder = str(dataset_configparser.get('Basics', 'out_folder'))
pulsemap_key = str(dataset_configparser.get('Basics', 'PulseSeriesMap'))


def save_to_array(phy_frame):
    # Function that saves the requested values in a dict of lists
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
                reco_arr.append(eval(el[1].replace('(x)', '(phy_frame)')))
            except Exception:
                print('uuupus {}'.format(el[1]))
                return False
        if (wf is not None) and (pulses is not None):
            events['waveforms'].append(wf)
            events['pulses'].append(pulses)
            events['reco_vals'].append(reco_arr)
    return True


def produce_data_dict(i3_file, geo_file):
    # IceTray script that wraps around an i3file and fills the events dict that is initialized outside the function
    tray = I3Tray()
    tray.AddModule("I3Reader", "source",
                   Filenamelist=[geo_file,
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
    tray.Execute()
    tray.Finish()
    return


def cuts(phy_event):
    cuts = dataset_configparser['Cuts']
    #### Checking for wierd event structures
    if testing_event(phy_event, geometry_file) == -1:
        report = [RunID, EventID, "EventTestingFailed"]
        return False
    ParticelList = [12, 14, 16]
    if cuts['only_neutrino_as_primary_cut'] == "ON":
        if abs(phy_event['MCPrimary'].pdg_encoding) not in ParticelList:
            report = [RunID, EventID, "NeutrinoPrimaryCut"]
            return False
    if cuts['max_energy_cut'] == "ON":
        energy_cutoff = cuts['max_energy_cutoff']
        if calc_depositedE(phy_event) > energy_cutoff:
            report = [RunID, EventID, "MaximalEnergyCut"]
            return False
    if cuts['minimal_tau_energy'] == "ON":
        I3Tree = phy['I3MCTree']
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
            if calc_depositedE(phy_event) < minimal_tau_energy:
                report = [RunID, EventID, "MinimalTauEnergyCut"]
                return False
    if cuts['min_energy_cut'] == "ON":
        energy_cutoff = int(cuts['min_energy_cutoff'])
        if calc_depositedE(phy_event) < energy_cutoff:
            report = [RunID, EventID, "MinimalEnergyCut"]
            return False
    if cuts['min_hit_DOMs_cut'] == "ON":
        hit_DOMs_cutoff = cuts['min_hit_DOMs']
        if calc_hitDOMs(phy_event) < hit_DOMS_cutoff:
            report = [RunID, EventID, "HitDOMsCut"]
            return False
    return True

##########

if __name__ == "__main__":

    # Raw print arguments
    print"\n ############################################"
    print("You are running the script with arguments: ")
    for a in args.keys():
        print(str(a) + ": " + str(args[a]))
    print"############################################\n "

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

    # Read filelist and define outfile
    # spezial version fpr filelists that are txt, was implemented for testing
    if str(dataset_configparser.get('Basics', 'filelist_typ')) == "txt":
        if len(args['filelist']) > 1:
            filelist = []
            for i in xrange(len(args['filelist'])):
                a = []
                flist = open(args['filelist'][i], 'r')
                for line in flist:
                    a.append(line.rstrip())
                pickle.dump( a, open( "saveee.p", "wb" ) )
                a = pickle.load( open( "saveee.p", "rb" ) )
                filelist.append(a)
            outfile = args['filelist'][0].replace('.txt', '.h5') 
        
        elif args['filelist'] != None:
            filelist=[]
            flist = open(args['filelist'], 'r')
            for line in flist:
                filelist.append(line.rstrip())
            pickle.dump( filelist, open( "save.p", "wb" ) )
            filelist = pickle.load( open( "save.p", "rb" ) )
            outfile = args['filelist'].replace('.txt', '.h5')

        elif args['files'] != None:
            filelist = args['files']
            outfile = filelist[0].replace('.i3.bz2', '.h5')

    # default version, when using the submit script
    elif str(dataset_configparser.get('Basics', 'filelist_typ')) == "pickle":
        if len(args['filelist']) > 1:
            filelist=[]
            for i in xrange(len(args['filelist'])):
                a = pickle.load(open(args['filelist'][i], 'r'))
                filelist.append(a)
            #path of outfile could be changed to a new folder for a  better overview
            outfile = args['filelist'][0].replace('.pickle', '.h5')
            
        elif args['filelist'] != None:
            filelist = pickle.load(open(args['filelist'], 'r'))
            outfile = args['filelist'].replace('.pickle', '.h5')

    else:
        raise Exception('No input files given')

    if os.path.exists(outfile):
        os.remove(outfile)

    dtype, data_source = fu.read_variables(dataset_configparser)
    dtype_len = len(dtype)
    FILTERS = tables.Filters(complib='zlib', complevel=9)
    with tables.open_file(
        outfile, mode="w", title="Events for training the NN",
            filters=FILTERS) as h5file:
        charge = h5file.create_earray(
            h5file.root, 'charge', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Sum(Charges per Dom)")
        time_first = h5file.create_earray(
            h5file.root, 'time', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Times of the first pulse")
        time_spread = h5file.create_earray(
            h5file.root, 'time_spread', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time delay between first and last pulse")
        charge_first = h5file.create_earray(
            h5file.root, 'first_charge', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="amplitude of the first charge")
        av_time_charges = h5file.create_earray(
            h5file.root, 'av_time_charges', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Weighted time average (charges)")
        num_pulses = h5file.create_earray(
            h5file.root, 'num_pulses', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Number of pulses")
        time_quartercharge = h5file.create_earray(
            h5file.root, 'time_quartercharge', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title=" Time where quarter of the total charge was detected ")
        time_kurtosis = h5file.create_earray(
            h5file.root, 'time_kurtosis', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="kurtosis of the time distr. of the pulses")
        time_moment_2 = h5file.create_earray(
            h5file.root, 'time_moment_2', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="second moment of time")
        time_10pct = h5file.create_earray(
            h5file.root, 'time_10pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 10 percent of charge is deposited")
        time_20pct = h5file.create_earray(
            h5file.root, 'time_20pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 20 percent of charge is deposited")
        time_30pct = h5file.create_earray(
            h5file.root, 'time_30pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 30 percent of charge is deposited")
        time_40pct = h5file.create_earray(
            h5file.root, 'time_40pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 40 percent of charge is deposited")
        time_50pct = h5file.create_earray(
            h5file.root, 'time_50pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 50 percent of charge is deposited")
        time_60pct = h5file.create_earray(
            h5file.root, 'time_60pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 60 percent of charge is deposited")
        time_70pct = h5file.create_earray(
            h5file.root, 'time_70pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 70 percent of charge is deposited")
        time_80pct = h5file.create_earray(
            h5file.root, 'time_80pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 80 percent of charge is deposited")
        time_90pct = h5file.create_earray(
            h5file.root, 'time_90pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 90 percent of charge is deposited")
        time_100pct = h5file.create_earray(
            h5file.root, 'time_100pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 100 percent of charge is deposited")
        time_15pct = h5file.create_earray(
            h5file.root, 'time_15pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 15 percent of charge is deposited")
        time_25pct = h5file.create_earray(
            h5file.root, 'time_25pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 25 percent of charge is deposited")
        time_35pct = h5file.create_earray(
            h5file.root, 'time_35pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 35 percent of charge is deposited")
        time_45pct = h5file.create_earray(
            h5file.root, 'time_45pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 45 percent of charge is deposited")
        time_55pct = h5file.create_earray(
            h5file.root, 'time_55pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 55 percent of charge is deposited")
        time_65pct = h5file.create_earray(
            h5file.root, 'time_65pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 65 percent of charge is deposited")
        time_75pct = h5file.create_earray(
            h5file.root, 'time_75pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 75 percent of charge is deposited")
        time_85pct = h5file.create_earray(
            h5file.root, 'time_85pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 85 percent of charge is deposited")
        time_95pct = h5file.create_earray(
            h5file.root, 'time_95pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 95 percent of charge is deposited")
        time_05pct = h5file.create_earray(
            h5file.root, 'time_05pct', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Time at which 05 percent of charge is deposited")
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
            print "Time for {} Sets of {} I3-Files: {}".\
                format(statusInFilelist, len(args['filelist']), starttime - timestamp)
            events = dict()
            events['reco_vals'] = []
            events['waveforms'] = []
            events['pulses'] = []
            counterSim = 0
            while counterSim < len(args['filelist']):
                fu.produce_data_dict(filelist[counterSim][statusInFilelist],
                                      geometry_file, dataset_configparser )
                try:
                    fu.produce_data_dict(filelist[counterSim][statusInFilelist],
                                         geometry_file, dataset_configparser)
                    counterSim = counterSim + 1
                except Exception:
                    statusInFilelist += 1
                    continue
                statusInFilelist += 1
                # shuffeling of the files
                print(len(events['reco_vals']))
                num_events = len(events['reco_vals'])
                shuff = np.random.choice(num_events, num_events, replace=False)
                for i in shuff:
                    TotalEventCounter += 1
                    reco_arr = events['reco_vals'][i]
                    if not len(reco_arr) == dtype_len:
                        continue

                    charge_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_first_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_spread_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    charge_first_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    av_time_charges_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    num_pulses_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_moment_2_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_kurtosis_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_10pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_20pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_30pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_40pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_50pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_60pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_70pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_80pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_90pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_100pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))        
                    time_15pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_25pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_35pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_45pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_55pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_65pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_75pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_85pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_95pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_05pct_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))

                    pulses = events['pulses']
                    waveforms = events['waveforms']
                    final_dict = dict()
                    for omkey in pulses.keys():
                        charges = np.array([p.charge for p in pulses[omkey][:]])
                        times = np.array([p.time for p in pulses[omkey][:]])
                        waveform = waveforms[omkey]
                        widths = np.array([p.width for p in pulses[omkey][:]])
                        final_dict[(omkey.string, omkey.om)] = \
                            (np.sum(charges),
                             np.amin(times),
                             np.amax(times) - np.amin(times),
                             charges[0],
                             np.average(charges, weights=1. / widths),
                             np.average(times, weights=charges),
                             len(charges),
                             moment(times, moment=2),
                             skew(times),
                             wf_quantiles(waveform, 10),
                             wf_quantiles(waveform, 20),
                             wf_quantiles(waveform, 30),
                             wf_quantiles(waveform, 40),
                             wf_quantiles(waveform, 50),
                             wf_quantiles(waveform, 60),
                             wf_quantiles(waveform, 70),
                             wf_quantiles(waveform, 80),
                             wf_quantiles(waveform, 90),
                             wf_quantiles(waveform, 100),
                             wf_quantiles(waveform, 15),
                             wf_quantiles(waveform, 25),
                             wf_quantiles(waveform, 35),
                             wf_quantiles(waveform, 45),
                             wf_quantiles(waveform, 55),
                             wf_quantiles(waveform, 65),
                             wf_quantiles(waveform, 75),
                             wf_quantiles(waveform, 85),
                             wf_quantiles(waveform, 95),
                             wf_quantiles(waveform, 05)
                             )

                    for dom in DOM_list:
                        gpos = grid[dom]
                        if dom in final_dict:
                            charge_arr[0][gpos[0]][gpos[1]][gpos[2]][0] += \
                                final_dict[dom][0]
                            charge_first_arr[0][gpos[0]][gpos[1]][gpos[2]][0] += \
                                final_dict[dom][3]
                            time_spread_arr[0][gpos[0]][gpos[1]][gpos[2]][0] += \
                                final_dict[dom][2]
                            time_first_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                    final_dict[dom][1]
                            #av_charge_widths_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                            #        final_dict[dom][4]
                            av_time_charges_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                    final_dict[dom][5]
                            num_pulses_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                    final_dict[dom][6]
                            time_moment_2_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][7]
                            time_kurtosis_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][8]
                            time_10pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][9]
                            time_20pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][10]
                            time_30pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][11]
                            time_40pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][12]
                            time_50pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][13]
                            time_60pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][14]
                            time_70pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][15]
                            time_80pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][16]
                            time_90pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][17]
                            time_100pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][18]
                            time_15pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][19]
                            time_25pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][20]
                            time_35pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][21]
                            time_45pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][22]
                            time_55pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][23]
                            time_65pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][24]
                            time_75pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][25]
                            time_85pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][26]
                            time_95pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][27]
                            time_05pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][28]


                    charge.append(np.array(charge_arr))
                    charge_first.append(np.array(charge_first_arr))
                    time_spread.append(np.array(time_spread_arr))
                    time_first.append(np.array(time_first_arr))
                    av_time_charges.append(av_time_charges_arr)
                    num_pulses.append(num_pulses_arr)
                    time_moment_2.append(time_moment_2_arr)
                    time_kurtosis.append(time_kurtosis_arr)
                    time_10pct.append(time_10pct_arr)
                    time_20pct.append(time_20pct_arr)
                    time_30pct.append(time_30pct_arr)
                    time_40pct.append(time_40pct_arr)
                    time_50pct.append(time_50pct_arr)
                    time_60pct.append(time_60pct_arr)
                    time_70pct.append(time_70pct_arr)
                    time_80pct.append(time_80pct_arr)
                    time_90pct.append(time_90pct_arr)
                    time_100pct.append(time_100pct_arr)
                    time_15pct.append(time_15pct_arr)
                    time_25pct.append(time_25pct_arr)
                    time_35pct.append(time_35pct_arr)
                    time_45pct.append(time_45pct_arr)
                    time_55pct.append(time_55pct_arr)
                    time_65pct.append(time_65pct_arr)
                    time_75pct.append(time_75pct_arr)
                    time_85pct.append(time_85pct_arr)
                    time_95pct.append(time_95pct_arr)
                    time_05pct.append(time_05pct_arr)                    
                    reco_vals.append(np.array(reco_arr))

        charge.flush()
        time_first.flush()
        charge_first.flush()
        time_spread.flush()
        av_time_charges.flush()
        num_pulses.flush()
        time_moment_2.flush()
        time_kurtosis.flush()
        time_10pct.flush()
        time_20pct.flush()
        time_30pct.flush()
        time_40pct.flush()
        time_50pct.flush()
        time_60pct.flush()
        time_70pct.flush()
        time_80pct.flush()
        time_90pct.flush()
        time_100pct.flush()
        time_15pct.flush()
        time_25pct.flush()
        time_35pct.flush()
        time_45pct.flush()
        time_55pct.flush()
        time_65pct.flush()
        time_75pct.flush()
        time_85pct.flush()
        time_95pct.flush()
        time_05pct.flush()

        print reco_vals
        reco_vals.flush()
        print"\n ###########################################################"
        print('###### Run Summary ###########')
        print('Processed: {} Frames \n Skipped {} \ Frames with Attribute Error \n To high depoited Energy {}'.format(TotalEventCounter, skipped_frames, frameToHighDepositedEnergy))
        if dataset_configparser.get('Basics', 'onlyneutrinoasprimary') == "True":
            print "\n No Neutrino as Primary {}".format(framesNotNeutrinoPrimary)
        print "\n Frames with a I3MCTree Problem {}".format(TreeProblem)
        print"############################################################\n " 
        print "Script is at its END"
        h5file.root._v_attrs.len = TotalEventCounter
    h5file.close()
    fail_file.close()
