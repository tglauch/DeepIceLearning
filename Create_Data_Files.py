#!/usr/bin/env python
# coding: utf-8

from icecube import dataclasses, dataio, icetray
import os
import numpy as np
import itertools
import math
import tables
import argparse


def parseArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--project", help="The name for the Project", type=str ,default='numu_complete')
  parser.add_argument("--num_files", help="The name for the Project", type=int ,default=-1)
  parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')
    # Parse arguments
  args = parser.parse_args()

  return args

def make_grid_dict(input_shape, geometry):
    """Put the Icecube Geometry in a cubic grid. For each DOM calculate the corresponding grid position

    Arguments:
    input_shape : The shape of the grid (x,y,z)
    geometry : Geometry file containing the positions of the DOMs in the Detector
    """
    
    grid = dict()
    DOM_List = [i for i in geometry.keys() if i.om<61]
    xpos=[geo[i].position.x for i in DOM_List]
    ypos=[geo[i].position.y for i in DOM_List]
    zpos=[geo[i].position.z for i in DOM_List]
    xmin, xmax = np.min(xpos), np.max(xpos)
    delta_x = (xmax - xmin)/input_shape[0]
    ymin, ymax = np.min(ypos), np.max(ypos)
    delta_y = (ymax - ymin)/input_shape[1]
    zmin, zmax = np.min(zpos), np.max(zpos)
    delta_z = (zmax - zmin)/input_shape[2]
    dom_list_ret = []
    for i, odom in enumerate(DOM_List):
        dom_list_ret.append((odom.string, odom.om))
        grid[(odom.string, odom.om)] = (int(math.floor((xpos[i]-xmin)/delta_x)),
                      int(math.floor((ypos[i]-ymin)/delta_y)),
                      int(math.floor((zpos[i]-zmin)/delta_z))
                      )
    return grid, dom_list_ret

if __name__ == "__main__":

    # Raw print arguments
    args = parseArguments()
    print"\n ############################################"
    print("You are running the script with arguments: ")
    for a in args.__dict__:
      print(str(a) + ": " + str(args.__dict__[a]))
    print"############################################\n "

    project_name = args.__dict__['project']

    input_shape = [20,20,50]  ### hardcoded at the moment
    file_location = '/data/user/tglauch/ML_Reco/'




    folderpath = '/data/ana/PointSource/PS/IC86_2012/files/sim/2012/neutrino-generator/11069/00000-00999/'
    #folderpath = '../ba_code/data/'
    filelist = [ f_name for f_name in os.listdir(folderpath) if f_name[-6:]=='i3.bz2']

    geometry_file = '/data/sim/sim-new/downloads/GCD/GeoCalibDetectorStatus_2012.56063_V0.i3.gz'
    #geometry_file='../ba_code/data/GeoCalibDetectorStatus_2012.56063_V0.i3.gz'
    geometry = dataio.I3File(geometry_file)
    geo = geometry.pop_frame()['I3Geometry'].omgeo

    ######### Create HDF5 File ##########
    FILTERS = tables.Filters(complib='zlib', complevel=9)
    h5file = tables.open_file(os.path.join(file_location, 'training_data/{}.h5'.format(project_name)), 
        mode = "w", title = "Events for training the NN", filters=FILTERS)

    charge = h5file.create_earray(h5file.root, 'charge', 
        tables.Float64Atom(), (0, 1 ,input_shape[0]+1,input_shape[1]+1,input_shape[2]+1),
        title = "Charge Distribution")
    time = h5file.create_earray(h5file.root, 'time', 
        tables.Float64Atom(), (0, 1,input_shape[0]+1,input_shape[1]+1,input_shape[2]+1), 
        title = "Timestamp Distribution")
    reco_vals = h5file.create_earray(h5file.root, 'reco_vals', tables.Float64Atom(), 
        (0,4),title = "Energy,Azimuth,Zenith,MuEx")

    print('Created a new HDF File with the Settings:')
    print(h5file)

    grid, DOM_list = make_grid_dict(input_shape,geo)
    #np.save('grid.npy', grid)
    j=0
    basepath = '/data/ana/PointSource/PS/IC86_2012/files/sim/2012/neutrino-generator/'
    folders = ['11029', '11069']
    #folders = ['11069/00000-00999/']
    #folderpath = '../ba_code/data/'
    for folder in folders:
        subfolders = os.listdir(os.path.join(basepath,folder))
        print('Process Folder: {}'.format(os.path.join(basepath,folder)))
        for subfolder in subfolders:
            filelist = [ f_name for f_name in os.listdir(os.path.join(basepath,folder,subfolder)) if f_name[-6:]=='i3.bz2']
            for counter, in_file in enumerate(filelist):
                if counter > args.__dict__['num_files'] and not args.__dict__['num_files']==-1:
                    continue
                print('Processing File {}/{}'.format(counter, len(filelist)))
                event_file = dataio.I3File(os.path.join(basepath, folder, subfolder, in_file))
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

                    ######### The +1 is not optimal....probably reconsider 
                    charge_arr = np.zeros((1, 1,input_shape[0]+1,input_shape[1]+1,input_shape[2]+1))
                    time_arr = np.full((1, 1,input_shape[0]+1,input_shape[1]+1,input_shape[2]+1), np.inf)

                    ###############################################
                    pulses = physics_event['InIceDSTPulses'].apply(physics_event)
                    final_dict = dict()
                    for omkey in pulses.keys():
                            temp_time = []
                            temp_charge = []
                            t_zero = np.inf
                            for pulse in pulses[omkey]:
                                temp_time.append(pulse.time)
                                temp_charge.append(pulse.charge)
                            final_dict[(omkey.string, omkey.om)] = (np.sum(temp_charge), np.mean(temp_time))
                            if np.mean(temp_time)<t_zero:
                                t_zero = np.mean(temp_time)
                    for dom in DOM_list:
                        grid_pos = grid[dom]
                        if  dom in final_dict:
                            charge_arr[0][0][grid_pos[0]][grid_pos[1]][grid_pos[2]] += final_dict[dom][0]
                            time_arr[0][0][grid_pos[0]][grid_pos[1]][grid_pos[2]] += final_dict[dom][1]

                    charge.append(np.array(charge_arr))
                    time.append(np.array(time_arr))
                    reco_vals.append(np.array([energy,azmiuth, zenith, muex])[np.newaxis])
                    j+=1
            charge.flush()
            time.flush()
            reco_vals.flush()

    h5file.close()
    # np.save('truevals.npy', np.array(trueval_arr))
    # np.save('time.npy', np.array(time_arr))
    # np.save('charge.npy', np.array(charge_arr))