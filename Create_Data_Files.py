#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v2/icetray-start
#METAPROJECT /data/user/tglauch/Software/combo/build
# coding: utf-8

from icecube import dataclasses, dataio, icetray
import numpy as np
import itertools
import math
import tables   
import argparse
import os, sys

#### File paths #########
basepath = '/data/ana/PointSource/PS/IC86_2012/files/sim/2012/neutrino-generator/'
geometry_file = '/data/sim/sim-new/downloads/GCD/GeoCalibDetectorStatus_2012.56063_V0.i3.gz'
file_location = '/data/user/tglauch/ML_Reco/'


input_shape = [20,10,60]  ### hardcoded at the moment


def parseArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--project", help="The name for the Project", type=str ,default='none')
  parser.add_argument("--num_files", help="The name for the Project", type=str ,default=-1)
  parser.add_argument("--folder", help="neutrino-generator folder", type=str ,default='11069/00000-00999')
  ### relative to /data/ana/PointSource/PS/IC86_2012/files/sim/2012/neutrino-generator/ , 
  ## use ':' seperator for more then one folder
  parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')
    # Parse arguments
  args = parser.parse_args()

  return args

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
    
    
    dom_6_pos = geometry[icetray.OMKey(6,1)].position
    dom_1_pos = geometry[icetray.OMKey(1,1)].position
    theta = -np.arctan( (dom_6_pos.y - dom_1_pos.y)/(dom_6_pos.x - dom_1_pos.x) )
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.matrix([[c, -s], [s, c]])
    
    grid = dict()
    DOM_List = [i for i in geometry.keys() if  i.om < 61                      # om > 60 are icetops
                                           and i.string not in range(79,87)]  # exclude deep core strings
    xpos=[geometry[i].position.x for i in DOM_List]
    ypos=[geometry[i].position.y for i in DOM_List]
    zpos=[geometry[i].position.z for i in DOM_List]
    
    rotxy = [np.squeeze(np.asarray(np.dot(rot_mat, xy))) for xy in zip(xpos, ypos)]
    xpos, ypos = zip(*rotxy)
    
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

def calc_depositedE(physics_frame):
    I3Tree = physics_frame['I3MCTree']
    truncated_energy = 0
    for i in I3Tree:
        interaction_type = str(i.type)
        if interaction_type in ['DeltaE','PairProd','Brems','EMinus']:
            truncated_energy += i.energy
    return truncated_energy

if __name__ == "__main__":

    # Raw print arguments
    args = parseArguments()
    print"\n ############################################"
    print("You are running the script with arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))
    print"############################################\n "

    if args.__dict__['project'] == 'none':
        project_name = (args.__dict__['folder']).replace('/','_').replace(':','_')
    else:
        project_name = args.__dict__['project']

    geometry = dataio.I3File(geometry_file)
    geo = geometry.pop_frame()['I3Geometry'].omgeo
    grid, DOM_list = make_grid_dict(input_shape,geo)

    ######### Create HDF5 File ##########
    OUTFILE = os.path.join(file_location, 'training_data/{}.h5'.format(project_name))
    if os.path.exists(OUTFILE):
        os.remove(OUTFILE)

    FILTERS = tables.Filters(complib='zlib', complevel=9)
    with tables.open_file(OUTFILE, mode = "w", title = "Events for training the NN", filters=FILTERS) as h5file:

        charge = h5file.create_earray(h5file.root, 'charge', 
            tables.Float64Atom(), (0, ,input_shape[0],input_shape[1],input_shape[2],1),
            title = "Charge Distribution")
        time = h5file.create_earray(h5file.root, 'time', 
            tables.Float64Atom(), (0, input_shape[0],input_shape[1],input_shape[2],1 ), 
            title = "Timestamp Distribution")
        reco_vals = h5file.create_earray(h5file.root, 'reco_vals', tables.Float64Atom(), 
            (0,6), title = "Energy,Azimuth,Zenith,MuEx,depositedE,ow")
    
        print('Created a new HDF File with the Settings:')
        print(h5file)

        np.save('grid.npy', grid)
        j=0
        folders = args.__dict__['folder'].split(':')
        for folder in folders:
            print('Process Folder: {}'.format(os.path.join(basepath,folder)))
            filelist = [ f_name for f_name in os.listdir(os.path.join(basepath,folder)) if f_name[-6:]=='i3.bz2']
            for counter, in_file in enumerate(filelist):
                if counter > int(args.__dict__['num_files']) and not int(args.__dict__['num_files'])==-1:
                    continue
                if counter%10 == 0 :
                    print('Processing File {}/{}'.format(counter, len(filelist)))
                event_file = dataio.I3File(os.path.join(basepath, folder, in_file))
                while event_file.more():
                    try:
                        physics_event = event_file.pop_physics()
                        energy = physics_event['MCMostEnergeticTrack'].energy
                    except AttributeError:
                        print('Attribute Error occured')
                        continue
                    if energy<100:
                        continue
                    azmiuth = physics_event['MCMostEnergeticTrack'].dir.azimuth 
                    zenith = physics_event['MCMostEnergeticTrack'].dir.zenith
                    muex = physics_event['SplineMPEMuEXDifferential'].energy
                    ow = physics_event['I3MCWeightDict']['OneWeight']
                    depositedE = calc_depositedE(physics_event)
                    charge_arr = np.zeros((1, input_shape[0],input_shape[1],input_shape[2], 1))
                    time_arr = np.full((1, input_shape[0],input_shape[1],input_shape[2], 1), np.inf)

                    ###############################################
                    pulses = physics_event['InIceDSTPulses'].apply(physics_event)
                    final_dict = dict()
                    for omkey in pulses.keys():
                            temp_time = []
                            temp_charge = []
                            for pulse in pulses[omkey]:
                                temp_time.append(pulse.time)
                                temp_charge.append(pulse.charge)
                            final_dict[(omkey.string, omkey.om)] = (np.sum(temp_charge), np.min(temp_time))
                    for dom in DOM_list:
                        grid_pos = grid[dom]
                        if dom in final_dict:
                            charge_arr[0][grid_pos[0]][grid_pos[1]][grid_pos[2]][0] += final_dict[dom][0]
                            time_arr[0][grid_pos[0]][grid_pos[1]][grid_pos[2]][0] = \
                                np.minimum(time_arr[0][grid_pos[0]][grid_pos[1]][grid_pos[2]][0], final_dict[dom][1])

                    charge.append(np.array(charge_arr))
                    # normalize time on [0,1]. not hit bins will still carry np.inf as time value
                    time_np_arr = np.array(time_arr)
                    time_np_arr_max = np.max(time_np_arr[time_np_arr != np.inf])
                    time_np_arr_min = np.min(time_np_arr)
                    time.append((time_np_arr - time_np_arr_min) / (time_np_arr_max - time_np_arr_min))
                    reco_vals.append(np.array([energy,azmiuth, zenith, muex, depositedE, ow])[np.newaxis])
                    j+=1
            charge.flush()
            time.flush()
            reco_vals.flush()

        h5file.close()