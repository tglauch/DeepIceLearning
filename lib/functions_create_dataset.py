from icecube import dataclasses
import logging
import numpy as np
import icecube.MuonGun
from icecube import dataclasses, dataio, simclasses
import scipy.stats as st
from icecube.weighting.weighting import from_simprod

def calc_gen_ow(frame, gcdfile):
    soft = from_simprod(11029)
    hard_lowE = from_simprod(11069)
    hard_highE = from_simprod(11070)
    generator = soft+hard_highE+hard_lowE
    ow = generator(frame['MCPrimary1'].energy, frame['I3MCWeightDict']['PrimaryNeutrinoType'],
                   np.cos(frame['MCPrimary1'].dir.zenith))
    return ow

def get_t0(frame, puls_key='InIceDSTPulses'):
    pulses = frame[puls_key]
    pul = pulses.apply(frame)
    time = []
    charge = []
    for i in pul:
        for j in i[1]:
            charge.append(j.charge)
            time.append(j.time)
    return median(time, weights=charge)


def median(arr, weights=None):
    if weights is not None:
        weights = 1. * np.array(weights)
    else:
        weights = np.ones(len(arr))
    rv = st.rv_discrete(values=(arr, weights / weights.sum()))
    return rv.median()


def get_most_E_muon_info(frame):
    gcdfile = '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2013.56429_V0.i3.gz'
    surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding=0)

    e0_list = []
    edep_list = []
    particle_list = []
    for track in icecube.MuonGun.Track.harvest(frame['I3MCTree'], frame['MMCTrackList']):
        # Find distance to entrance and exit from sampling volume
        intersections = surface.intersection(track.pos, track.dir)
        # Get the corresponding energies
        e0, e1 = track.get_energy(intersections.first), track.get_energy(intersections.second)
        e0_list.append(e0)
        edep_list.append((e0-e1))
        particle_list.append(track)
        # Accumulate
    edep_list = np.array(edep_list)
    inds = np.argsort(edep_list)[::-1]
    e0_list = np.array(e0_list)[inds]
    particle_list = np.array(particle_list)[inds]
    frame.Put("Reconstructed_Muon", particle_list[0])
    frame.Put("mu_E_on_entry", dataclasses.I3Double(e0_list[0]))
    frame.Put("mu_E_deposited", dataclasses.I3Double(edep_list[inds][0]))
    return


def read_variables(cfg_parser):
    """Function reading a config file, defining the variables to be read
       from the MC files.

    Arguments:
    cfg_parser: config parser object for the config file

    Returns:
    dtype : the dtype object defining the shape and names of the MC output
    data_source: list defining the types,names and ranges of monte carlo data
                to be saved from a physics frame
                (e.g [('variable',['MCMostEnergeticTrack'].energy, [1e2,1e9])])
    """
    dtype = []
    data_source = []
    cut = [-np.inf, np.inf]
    for i in cfg_parser['Variables'].keys():
        data_source.append(('variable', cfg_parser['Variables'][i], cut))
        dtype.append((str(i), np.float64))
    for i in cfg_parser['Functions'].keys():
        data_source.append(('function', cfg_parser['Functions'][i]+'(_icframe_)', cut))
        dtype.append((str(i), np.float64))
    dtype = np.dtype(dtype)
    return dtype, data_source


def cuts(phy_frame):
    """Performe a pre-selection of events according
       to the cuts defined in the config file

    Args:
        phy_frame, and IceCube I3File
    Returns:
        True (IceTray standard)
    """

    if phy_frame['CorsikaWeightMap']['Multiplicity'] > 1.:
        print('Multiplicity > 1')
        return False
    else:
        return True


def get_stream(phy_frame):
    #time_stream_0 = time.time()
    if not phy_frame['I3EventHeader'].sub_event_stream == 'NullSplit':
        #print "Time Get Stream: {}".format(time.time() - time_stream_0) 
        return False
    else:
        #print "Time Get Stream: {}".format(time.time() - time_stream_0)
        return True
