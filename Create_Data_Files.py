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

def time_of_percentage(charges, times, percentage):
    charges = charges.tolist()
    cut = np.sum(charges)/(100/percentage)
    sum=0
    for i in charges:
        sum = sum + i
        if sum > cut:
            tim = times[charges.index(i)]
            break
    return tim

##########

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
            filelist=[]
            for i in xrange(len(args['filelist'])):
                a=[]
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
        charge_100ns = h5file.create_earray(
            h5file.root, 'charge_100ns', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Sum of the Charge during the first 100ns per DOM")
        charge_200ns = h5file.create_earray(
            h5file.root, 'charge_200ns', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Sum of the Charge during the first 200ns per DOM")
        charge_300ns = h5file.create_earray(
            h5file.root, 'charge_300ns', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Sum of the Charge during the first 300ns per DOM")
        charge_400ns = h5file.create_earray(
            h5file.root, 'charge_400ns', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Sum of the Charge during the first 400ns per DOM")
        charge_500ns = h5file.create_earray(
            h5file.root, 'charge_500ns', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Sum of the Charge during the first 500ns per DOM")
        charge_600ns = h5file.create_earray(
            h5file.root, 'charge_600ns', tables.Float64Atom(),
            (0, input_shape[0], input_shape[1], input_shape[2], 1),
            title="Sum of the Charge during the first 600ns per DOM")
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
        reco_vals = tables.Table(h5file.root, 'reco_vals',
                                 description=dtype)
        h5file.root._v_attrs.shape = input_shape
        print('Created a new HDF File with the Settings:')
        print(h5file)
        print(h5file.root)
    
        np.save('grid.npy', grid)
        TotalEventCounter = 0
        skipped_frames = 0
        TreeProblem = 0
        framesNotNeutrinoPrimary = 0
        frameToHighDepositedEnergy = 0
        statusInFilelist=0
        event_files = []
        #print "##################################"
        #print len(filelist[0])
        #print len(filelist[1])
        #print args['filelist']
        #print "##################################"
        starttime = time.time()
        while statusInFilelist < len(filelist[0]):
            timestamp = time.time()
            print "Time for {} Sets of {} I3-Files: {}".format(statusInFilelist ,len(args['filelist']), starttime-timestamp)
            event_files = []
            counterSim=0
            while counterSim < len(args['filelist']):
                try:
                    event_files.append(dataio.I3File(filelist[counterSim][statusInFilelist], "r"))
                    counterSim = counterSim+1
                except Exception:
                    statusInFilelist += 1       
                    continue
            statusInFilelist += 1
            # shuffeling of the files
            while not len(event_files) == 0: 
                TotalEventCounter +=1
                a=random.choice(event_files) 
                #eventsToProcess=random.randint(1, 4)
                if a.more():
                    try:
                        physics_event = a.pop_physics()
                    except Exception:
                        print "Frame not poped"
                        continue
                    #try to open the I3MCTree, if not possible skip frame
                    try:
                        I3Tree = physics_event['I3MCTree']
                    except Exception:
                        print "Problem with the I3MCTree"
                        TreeProblem +=1
                        continue
                    # Possibility to only choose events with neutrinos as primary
                    ParticelList = [12, 14, 16]
                    if dataset_configparser.get('Basics', 'onlyneutrinoasprimary') == "True":
                        if abs(int(eval('physics_event{}'.format(dataset_configparser.get('firstParticle', 'variable'))))) not in ParticelList:
                                framesNotNeutrinoPrimary +=1
                                continue
                    # Possibility to define a maximal energy
                    energy_cutoff = int(dataset_configparser.get('Basics', 'energy_cutoff'))
                    if calc_depositedE(physics_event) > energy_cutoff:
                        frameToHighDepositedEnergy +=1
                        continue

                    # Possibility to define a minimal energy requierment for taus only
                    #First we need the neutrino
                    primary_list = I3Tree.get_primaries()
                    if len(primary_list) == 1:
                        neutrino = I3Tree[0]
                    else:
                        for p in primary_list:
                            pdg = p.pdg_encoding
                            if abs(pdg) in ParticelList:
                                neutrino = p
                    minimal_tau_energy = int(dataset_configparser.get('Basics', 'minimal_tau_energy'))
                    if abs(neutrino.pdg_encoding) == 16:
                        if calc_depositedE(physics_event) < minimal_tau_energy:
                            continue 

                    reco_arr = []
                    for k, cur_var in enumerate(data_source):
                        if cur_var[0] == 'variable':
                            try:
                                cur_value = eval(
                                    'physics_event{}'.format(cur_var[1]))
                            except Exception:
                                skipped_frames += 1
                                print('Attribute Error occured :{}'.
                                      format(cur_var[1]))
                                break
                        if cur_var[0] == 'function':
                            try:
                                cur_value = eval(
                                    cur_var[1].replace('(x)', '(physics_event)'))
                            except Exception:
                                skipped_frames += 1
                                print(
                                    'The given function {} is not implemented'.format(cur_var[1]))
                                break
                        if cur_value < cur_var[2][0] or cur_value > cur_var[2][1]:
                            break
                        else:
                            reco_arr.append(cur_value)
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
                    #av_charge_widths_arr = np.zeros(
                    #    (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    av_time_charges_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    num_pulses_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_moment_2_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    time_kurtosis_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    charge_100ns_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    charge_200ns_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    charge_300ns_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    charge_400ns_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    charge_500ns_arr = np.zeros(
                        (1, input_shape[0], input_shape[1], input_shape[2], 1))
                    charge_600ns_arr = np.zeros(
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

        
                    pulses = physics_event[pulsemap_key].apply(physics_event)
                    final_dict = dict()
                    for omkey in pulses.keys():
                        charges = np.array([p.charge for p in pulses[omkey][:]])
                        times = np.array([p.time for p in pulses[omkey][:]])
                        #times_shifted = times-np.amin(times)
                        widths = np.array([p.width for p in pulses[omkey][:]])
                        final_dict[(omkey.string, omkey.om)] = \
                            (np.sum(charges),
                             np.amin(times),
                             np.amax(times) - np.amin(times),
                             charges[0],\
                             np.average(charges,weights=1./widths),\
                             np.average(times, weights=charges),\
                             len(charges),\
                             moment(times, moment=2),\
                             skew(times),\
                             np.sum(charges[times<100]),\
                             np.sum(charges[times<200]),\
                             np.sum(charges[times<300]),\
                             np.sum(charges[times<400]),\
                             np.sum(charges[times<500]),\
                             np.sum(charges[times<600]),\
                             time_of_percentage(charges, times, 10),\
                             time_of_percentage(charges, times, 20),\
                             time_of_percentage(charges, times, 30),\
                             time_of_percentage(charges, times, 40),\
                             time_of_percentage(charges, times, 50),\
                             time_of_percentage(charges, times, 60),\
                             time_of_percentage(charges, times, 70),\
                             time_of_percentage(charges, times, 80),\
                             time_of_percentage(charges, times, 90),\
                             time_of_percentage(charges, times, 100)
                             )
                    #print "Checkpoint B"
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
                            charge_100ns_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][9]
                            charge_200ns_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][10]
                            charge_300ns_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom[11]
                            charge_400ns_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][12]
                            charge_500ns_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][13]
                            charge_600ns_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][14]
                            time_10pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][15]
                            time_20pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][16]
                            time_30pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][17]
                            time_40pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][18]
                            time_50pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][19]
                            time_60pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][20]
                            time_70pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][21]
                            time_80pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][22]
                            time_90pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][23]
                            time_100pct_arr[0][gpos[0]][gpos[1]][gpos[2]][0] = \
                                        final_dict[dom][24]



                    charge.append(np.array(charge_arr))
                    charge_first.append(np.array(charge_first_arr))
                    time_spread.append(np.array(time_spread_arr))
                    time_first.append(np.array(time_first_arr))
                    #av_charge_widths.append(av_charge_widths_arr)
                    av_time_charges.append(av_time_charges_arr)
                    num_pulses.append(num_pulses_arr)
                    time_moment_2.append(time_moment_2_arr)
                    time_kurtosis.append(time_kurtosis_arr)
                    charge_100ns.append(charge_100ns_arr)
                    charge_200ns.append(charge_200ns_arr)
                    charge_300ns.append(charge_300ns_arr)
                    charge_400ns.append(charge_400ns_arr)
                    charge_500ns.append(charge_500ns_arr)
                    charge_600ns.append(charge_600ns_arr)
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
                       
                    reco_vals.append(np.array(reco_arr))
                    #print "End Point"
                    #EventCounter +=1
                else:
                    print "no more in a"
                    event_files.remove(a)
                #TotalEventCounter = TotalEventCounter + EventCounter


        charge.flush()
        time_first.flush()
        charge_first.flush()
        time_spread.flush()
        #av_charge_widths.flush()
        av_time_charges.flush()
        num_pulses.flush()
        time_moment_2.flush()
        time_kurtosis.flush()
        charge_100ns.flush()
        charge_200ns.flush()
        charge_300ns.flush()
        charge_400ns.flush()
        charge_500ns.flush()
        charge_600ns.flush()
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
