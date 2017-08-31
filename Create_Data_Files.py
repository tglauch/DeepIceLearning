#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v2/icetray-start
#METAPROJECT /data/user/tglauch/Software/combo/build
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

from icecube import dataclasses, dataio, icetray
import numpy as np
import math
import tables 
import argparse
import os, sys
from configparser import ConfigParser
from reco_quantities import *


def parseArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_config", help="main config file, user-specific",\
                      type=str ,default='default.cfg')
  parser.add_argument("--files", help="files to be processed",
                      type=str, nargs="+", required=False)
  parser.add_argument("--filelist", help="Path to filelists to be processed",
                      type=str, nargs="+", required=False)
  parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')
  args = parser.parse_args()

  return args

args = parseArguments()
dataset_configparser = ConfigParser()
try:
    dataset_configparser.read(args.dataset_config)
except:
    raise Exception('Config File is missing!!!!') 


basepath = str(dataset_configparser.get('Basics', 'MC_path'))
geometry_file = str(dataset_configparser.get('Basics', 'geometry_file'))
outfolder = str(dataset_configparser.get('Basics', 'out_folder'))
mc_folder = dataset_configparser.get('Basics', 'folder')
pulsemap_key = str(dataset_configparser.get('Basics', 'PulseSeriesMap'))
file_list = args.files


def read_variables(cfg_parser):
    """Function reading a config file, defining the variables to be read from the MC files.

    Arguments:
    cfg_parser: config parser object for the config file
    
    Returns:
    dtype : the dtype object defining the shape and names of the MC output
    data_source: list defining the types,names and ranges of monte carlo data 
                to be saved from a physics frame (e.g [('variable',['MCMostEnergeticTrack'].energy, [1e2,1e9])])
    """
    dtype = []
    data_source = []
    for i, key in enumerate(cfg_parser.keys()):
        if key == 'DEFAULT' or key =='Basics':
            continue
        cut = [-np.inf, np.inf]
        if 'min' in cfg_parser[key].keys():
            cut[0] = float(cfg_parser[key]['min'])
        if 'max' in cfg_parser[key].keys():
            cut[1] = float(cfg_parser[key]['max'])  
        if 'variable' in cfg_parser[key].keys():
            data_source.append(('variable', cfg_parser[key]['variable'], cut))
        elif 'function' in cfg_parser[key].keys():
            data_source.append(('function', cfg_parser[key]['function'], cut))
        else:
            raise Exception('No Input Type given. Variable or funtion must be given')        
        dtype.append((str(key), eval('np.'+cfg_parser[key]['out_type'])))
    dtype=np.dtype(dtype)

    return dtype, data_source

def preprocess_grid(geometry):
    ## rotate IC into x-y-plane
    dom_6_pos = geometry[icetray.OMKey(6,1)].position
    dom_1_pos = geometry[icetray.OMKey(1,1)].position
    theta = -np.arctan( (dom_6_pos.y - dom_1_pos.y)/(dom_6_pos.x - dom_1_pos.x) )
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.matrix([[c, -s], [s, c]])

    DOM_List = [i for i in geometry.keys() if  i.om < 61                      # om > 60 are icetops
                                           and i.string not in range(79,87)]  # exclude deep core strings
    xpos=[geometry[i].position.x for i in DOM_List]
    ypos=[geometry[i].position.y for i in DOM_List]
    zpos=[geometry[i].position.z for i in DOM_List]

    rotxy = [np.squeeze(np.asarray(np.dot(rot_mat, xy))) for xy in zip(xpos, ypos)]
    xpos, ypos = zip(*rotxy)
    return xpos, ypos, zpos, DOM_List

def make_grid_dict(input_shape, geometry):
    """Put the Icecube Geometry in a cubic grid. For each DOM calculate the corresponding grid position. Rotates the x-y-plane
    in order to make icecube better fit into a grid.

    Arguments:
    input_shape : The shape of the grid (x,y,z)
    geometry : Geometry file containing the positions of the DOMs in the Detector

    Returns:
    grid: a dictionary mapping (string, om) => (grid_x, grid_y, grid_z), i.e. dom id to its index position in the cubic grid
    dom_list_ret: list of all (string, om), i.e. list of all dom ids in the geofile  (sorted(dom_list_ret)==sorted(grid.keys()))
    """ 
    grid = dict()
    xpos, ypos, zpos, DOM_List = preprocess_grid(geometry)

    xmin, xmax = np.min(xpos), np.max(xpos)
    delta_x = (xmax - xmin)/(input_shape[0]-1)
    xmin, xmaz = xmin - delta_x/2, xmax + delta_x/2
    ymin, ymax = np.min(ypos), np.max(ypos)
    delta_y = (ymax - ymin)/(input_shape[1]-1)
    ymin, ymaz = ymin - delta_y/2, ymax + delta_y/2
    zmin, zmax = np.min(zpos), np.max(zpos)
    delta_z = (zmax - zmin)/(input_shape[2]-1)
    zmin, zmax = zmin - delta_z/2, zmax + delta_z/2
    dom_list_ret = []
    for i, odom in enumerate(DOM_List):
        dom_list_ret.append((odom.string, odom.om))
        # for all x,y,z-positions the according grid position is calculated and stored.
        # the last items (i.e. xmax, ymax, zmax) are put in the last bin. i.e. grid["om with x=xmax"]=(input_shape[0]-1,...)
        grid[(odom.string, odom.om)] = (min(int(math.floor((xpos[i]-xmin)/delta_x)),
                                            input_shape[0]-1
                                           ),
                                        min(int(math.floor((ypos[i]-ymin)/delta_y)),
                                            input_shape[1]-1
                                           ),
                                        input_shape[2] - 1 -
                                            min(int(math.floor((zpos[i]-zmin)/delta_z)),
                                                input_shape[2]-1
                                           ) # so that z coordinates count from bottom to top (righthanded coordinate system)
                                       )
    return grid, dom_list_ret


def make_autoHexGrid(geometry):
    """Put the Icecube Geometry in a rectangular grid. For each DOM calculate the corresponding grid position. Rotates the x-y-plane
    in order to make icecube better fit into a grid.
    Method: aligns IC-strings which are not on the hexagonal grid + shifts
    x_positions such that no unfilled holes appear in the grid but rather empty
    edges (reduces dimensionality of the input and makes pattern recognition
    much easier)

    Arguments:
    geometry : Geometry file containing the positions of the DOMs in the Detector

    Returns:
    grid: a dictionary mapping (string, om) => (grid_x, grid_y, grid_z), i.e. dom id to its index position in the cubic grid
    dom_list_ret: list of all (string, om), i.e. list of all dom ids in the geofile  (sorted(dom_list_ret)==sorted(grid.keys()))
    """

    grid = dict()
    ## assumes the standard IC shape:
    max_string = max(o.string for o in geometry.keys())
    max_dom = max(o.om for o in geometry.keys())
    if max_string<78 or max_dom <60:
        print "Define your own input_shape, makeHexGrid is only for standardIC"
        raise NameError('Wrong geometry file for standard IC processing')

    xpos, ypos, zpos, DOM_List = preprocess_grid(geometry)
    deltax = abs(xpos[0]-xpos[60]) # inserted by hand, any better idea ? 
    deltay = abs(ypos[360]-ypos[0])

    nxRows, nyRows = 20, 10 ## again, standard IC geometry (20x10 grid with holes)
    ## align strings which do not lie on the hexagonal grid:
    xBands = np.linspace(np.amin(xpos)-deltax/4., np.amax(xpos)+deltax/4.,
                         nxRows+1)
    yBands = np.linspace(np.amin(ypos)-deltay/2., np.amax(ypos)+deltay/2.,
                         nyRows+1)
    xIndcs = np.digitize(xpos, xBands)
    yIndcs = np.digitize(ypos, yBands)
    # reset positions to the exact hex-grid positions
    xpos_aligned = deltax/4.*xIndcs
    ypos_aligned = deltay/2.*yIndcs

    # update deltas
    deltax_aligned = abs(xpos_aligned[0]-xpos_aligned[60])
    deltay_aligned = abs(ypos_aligned[360]-ypos_aligned[0])

    ## shift the x-positions of each DOM to shift the hex-grid to a rect-grid
    xpos_shifted = xpos_aligned + deltax_aligned/2. *\
            np.floor((ypos_aligned-(np.amin(ypos_aligned)+1e-5))/deltay_aligned)
    ## center the new grid
    x_final = xpos_shifted - np.mean(xpos_shifted)
    y_final = ypos_aligned - np.mean(xpos_aligned)

    ## final grid: 
    xinput_bins = np.linspace(np.amin(x_final)-deltax_aligned/2.,\
                              np.amax(x_final)+deltax_aligned/2.,\
                              12)
    yinput_bins = np.linspace(np.amin(y_final)-deltay_aligned/2.,\
                              np.amax(y_final)+deltay_aligned/2.,\
                              11)
    zinput_bins = np.linspace(np.amin(zpos),np.amax(zpos),60)

    dom_list_ret = []
    for i, odom in enumerate(DOM_List):
        dom_list_ret.append((odom.string, odom.om))
        grid[(odom.string, odom.om)] = (np.digitize([x_final[i]], xinput_bins)[0],
                                        np.digitize([y_final[i]], yinput_bins)[0],
                                        np.digitize([zpos[i]], zinput_bins)[0])
    return grid, dom_list_ret



def analyze_grid(grid):
    """
    if you want to see which string/om the bins contain
    """
    dims = []
    for dim in range(3):
        for index in range(input_shape[dim]):
            strings=set()
            dims.append(list())
            for k, v in grid.items():
                if v[dim] == index:
                    if dim == 2:
                        strings.add(k[1]) ## print om
                    else:
                        strings.add(k[0]) ## print string
            dims[dim].append(strings)
    for i, c in enumerate("xyz"):
        print c
        for index, strings in enumerate(dims[i]):
            print index, strings

if __name__ == "__main__":

    # Raw print arguments
    print"\n ############################################"
    print("You are running the script with arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))
    print"############################################\n "

    geo = dataio.I3File(geometry_file).pop_frame()['I3Geometry'].omgeo
    
    input_shape_par = dataset_configparser.get('Basics', 'input_shape')
    if input_shape_par != "auto":
        input_shape = eval(input_shape_par)
        grid, DOM_list = make_grid_dict(input_shape, geo)
    else:
        input_shape = [12,11,61]
        grid, DOM_list = make_autoHexGrid(geo)

    ######### Create HDF5 File ##########

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    if 'filelist' in args.keys():
        filelist = []
        with open(args['filelist'], 'r') as f:
            for line in f.readlines():
                filelist.append(line[:-2])
        outfile = args['filelist'].replace('.txt', '.npy') 
    elif 'files' in args.keys():
        filelist = args['files']
        outfile = filelist[0].replace('.i3.bz2', '.npy') 
    else:
        raise Exception('No input files given')

    if os.path.exists(OUTFILE):
        os.remove(OUTFILE)

    dtype, data_source = read_variables(dataset_configparser)
    dtype_len = len(dtype)
    FILTERS = tables.Filters(complib='zlib', complevel=9)
    with tables.open_file(OUTFILE, mode = "w", title = "Events for training the NN", filters=FILTERS) as h5file:

        charge = h5file.create_earray(h5file.root, 'charge',
            tables.Float64Atom(), (0, input_shape[0],input_shape[1],input_shape[2],1),
            title = "Sum(Charge)")
        time_first = h5file.create_earray(h5file.root, 'time',
            tables.Float64Atom(), (0, input_shape[0],input_shape[1],input_shape[2],1),
            title = "Times of the first pulse")
        time_spread = h5file.create_earray(h5file.root, 'time_spread',
            tables.Float64Atom(), (0, input_shape[0],input_shape[1],input_shape[2],1),
            title = "Time delay between first and last pulse")
        charge_first = h5file.create_earray(h5file.root, 'first_charge',
            tables.Float64Atom(), (0, input_shape[0],input_shape[1],input_shape[2],1),
            title = "amplitude of the first charge")
        reco_vals = tables.Table(h5file.root, 'reco_vals', description = dtype)
        h5file.root._v_attrs.shape = input_shape
        print('Created a new HDF File with the Settings:')
        print(h5file)

        np.save('grid.npy', grid)
        j=0
        skipped_frames = 0
        for counter, f_name in enumerate(filelist):
            if counter%10 == 0 :
                print('Processing File {}/{}'.format(counter, len(file_list)))
            event_file = dataio.I3File(f_name, "r")
            print "Opening succesful"
            while event_file.more():
                physics_event = event_file.pop_physics()
                reco_arr = []
                for k, cur_var in enumerate(data_source):
                    if cur_var[0]=='variable':
                        try:
                            cur_value = eval('physics_event{}'.format(cur_var[1]))
                        except:
                            skipped_frames += 1
                            print('Attribute Error occured')
                            break

                    if cur_var[0]=='function':
                        try:
                            cur_value = eval(cur_var[1].replace('(x)', '(physics_event)'))
                        except:
                            skipped_frames += 1
                            print('The given function seems to be not implemented')
                            break

                    if cur_value<cur_var[2][0] or cur_value>cur_var[2][1]:
                        break
                    else:
                        reco_arr.append(cur_value)

                if not len(reco_arr) == dtype_len:
                    continue
                charge_arr = np.zeros((1, input_shape[0],input_shape[1],input_shape[2], 1))
                time_first_arr = np.zeros((1, input_shape[0],input_shape[1],input_shape[2], 1))
                time_spread_arr = np.zeros((1, input_shape[0],input_shape[1],input_shape[2], 1))
                charge_first_arr = np.zeros((1, input_shape[0],input_shape[1],input_shape[2], 1))

                ###############################################
                pulses = physics_event[pulsemap_key].apply(physics_event)
                final_dict = dict()
                for omkey in pulses.keys():
                        temp_time = []
                        temp_charge = []
                        for pulse in pulses[omkey]:
                            temp_time.append(pulse.time)
                            temp_charge.append(pulse.charge)
                        final_dict[(omkey.string, omkey.om)] = (np.sum(temp_charge),\
                                                                np.min(temp_time),\
                                                                np.max(temp_time)-np.min(temp_time),\
                                                                temp_charge[0])
                for dom in DOM_list:
                    grid_pos = grid[dom]
                    if dom in final_dict:
                        charge_arr[0][grid_pos[0]][grid_pos[1]][grid_pos[2]][0]+=\
                                final_dict[dom][0]
                        charge_first_arr[0][grid_pos[0]][grid_pos[1]][grid_pos[2]][0]+=\
                                final_dict[dom][3]
                        charge_arr[0][grid_pos[0]][grid_pos[1]][grid_pos[2]][0]+=\
                                final_dict[dom][2]
                        time_first_arr[0][grid_pos[0]][grid_pos[1]][grid_pos[2]][0] = \
                            np.minimum(time_first_arr[0][grid_pos[0]][grid_pos[1]][grid_pos[2]][0], final_dict[dom][1])

                charge.append(np.array(charge_arr))
                charge_first.append(np.array(charge_first_arr))
                time_spread.append(np.array(time_spread_arr))
                time_np_arr = np.array(time_first_arr)
                #time_np_arr_max = np.max(time_np_arr[time_np_arr != np.inf])
                #time_np_arr_min = np.min(time_np_arr)
                #normalize time on [0,1]. not hit bins will still carry np.inf as time value
                #time.append((time_np_arr - time_np_arr_min) / (time_np_arr_max - time_np_arr_min))
                time_first.append(time_np_arr)
                reco_vals.append(np.array(reco_arr))
                j+=1
            charge.flush()
            time_first.flush()
            charge_first.flush()
            time_spread.flush()
            reco_vals.flush()
        print('###### Run Summary ###########')
        print('Processed: {} Frames \n Skipped {} Frames with Attribute Error'.format(j,skipped_frames))
        h5file.root._v_attrs.len = j
        h5file.close()
