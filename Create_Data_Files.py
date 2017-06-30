
# coding: utf-8

from icecube import dataclasses, dataio, icetray
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

input_shape = [20,20,50]

def make_grid_dict(input_shape, geometry):
    """Put the Icecube Geometry in a cubic grid. For each DOM calculate the grid position

    Arguments:
    input_shape : The shape of the grid (x,y,z)
    geometry : Geometry file containing the positions of the DOMs in the Detector
    """
    
    grid = dict()
    DOM_List = geometry.keys()
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


folderpath = '/data/ana/PointSource/PS/IC86_2012/files/sim/2012/neutrino-generator/11069/00000-00999/'
#folderpath = '../ba_code/data/'
filelist = [ f_name for f_name in os.listdir(folderpath) if f_name[-6:]=='i3.bz2']

geometry_file='/data/sim/sim-new/downloads/GCD/GeoCalibDetectorStatus_2012.56063_V0.i3.gz'
#geometry_file='../ba_code/data/GeoCalibDetectorStatus_2012.56063_V0.i3.gz'
geometry = dataio.I3File(geometry_file)
geo = geometry.pop_frame()['I3Geometry'].omgeo

######### New Version ##########
grid, DOM_list = make_grid_dict(input_shape,geo)
j=0
charge_arr = []
time_arr = []
trueval_arr = []
for counter, in_file in enumerate(filelist):
    if counter > 100:
        continue
    print counter
    event_file = dataio.I3File(os.path.join(folderpath, in_file))
    while event_file.more():
            ######### The +1 is not optimel....probably reconsider 
            charge_arr.append(np.zeros((input_shape[0]+1,input_shape[1]+1,input_shape[2]+1)))
            time_arr.append(np.full((input_shape[0]+1,input_shape[1]+1,input_shape[2]+1), np.inf))
            ###############################################
            physics_event = event_file.pop_physics()
            pulses = physics_event['InIceDSTPulses'].apply(physics_event)
            final_dict = dict()
            for omkey in pulses.keys():
                    temp_time = []
                    temp_charge = []
                    t_zero = np.inf
                    for pulse in pulses[omkey]:
                        temp_time.append(pulse.time)
                        temp_charge.append(pulse.charge)
                    final_dict[(omkey.string, omkey.om)] = (np.mean(temp_charge) ,np.mean(temp_time))
                    if np.mean(temp_time)<t_zero:
                        t_zero = np.mean(temp_time)
            for dom in DOM_list:
                grid_pos = grid[dom]
                if  dom in final_dict:
                    charge_arr[j][grid_pos[0]][grid_pos[1]][grid_pos[2]] += final_dict[dom][0]
                    time_arr[j][grid_pos[0]][grid_pos[1]][grid_pos[2]] += final_dict[dom][1]
            energy = physics_event['MCMostEnergeticTrack'].energy
            azmiuth = physics_event['MCMostEnergeticTrack'].dir.azimuth 
            zenith = physics_event['MCMostEnergeticTrack'].dir.zenith
            trueval_arr.append([energy,azmiuth, zenith])
            j+=1

np.save('truevals.npy', np.array(trueval_arr))
np.save('time.npy', np.array(time_arr))
np.save('charge.npy', np.array(charge_arr))