from icecube import dataio, icetray, phys_services
from icecube import dataclasses, paraboloid, simclasses, recclasses, spline_reco
from I3Tray import *
import sys
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.join(os.path.abspath(".."),'lib'))
import lib.reco_quantities as reco_q
from lib.functions_create_dataset import get_t0
import numpy as np
from icecube.weighting import weighting, get_weighted_primary
import icecube.MuonGun
sys.path.append('/data/user/tglauch/DeepIceLearning/I3Module')
#from i3module import DeepLearningClassifier
from icecube import DomTools, linefit
from icecube import gulliver,paraboloid,lilliput
#from icecube import NewNuFlux
from icecube.icetray import I3Units

def cuts2(phy_frame):
    """Performe a pre-selection of events according
       to the cuts defined in the config file

    Args:
        phy_frame, and IceCube I3File
    Returns:
        True (IceTray standard)
    """

    if 'CorsikaWeightMap' in phy_frame:
        if (phy_frame['CorsikaWeightMap']['Multiplicity'] > 1):
            return False
    elif 'PolyplopiaCount' in phy_frame:
        if (phy_frame['PolyplopiaCount'].value > 0): 
            return False

    if (phy_frame['MCPrimary1'].dir.zenith<np.radians(80)) :
        return False
    else:
        return True

def cuts(phy_frame):
    """Performe a pre-selection of events according
       to the cuts defined in the config file

    Args:
        phy_frame, and IceCube I3File
    Returns:
        True (IceTray standard)
    """
    
    if "mu_E_on_entry" not in phy_frame:
        return False

    if (phy_frame['mu_E_on_entry'].value<10):
        return False
    return True


def print_info(phy_frame):
    if reco_q.is_data(phy_frame):
        return True
    print('run_id {} ev_id {} dep_E {} classification {}  signature {} track_length {}'.format(
          phy_frame['I3EventHeader'].run_id, phy_frame['I3EventHeader'].event_id, 
          phy_frame['depE'].value, phy_frame['classification'].value,
          phy_frame['signature'].value, phy_frame['track_length'].value))
    return

def print_short(phy_frame):
    print('run_id {} ev_id {} '.format(
          phy_frame['I3EventHeader'].run_id, phy_frame['I3EventHeader'].event_id))
    return

def add_weighted_primary(phy_frame):
    if reco_q.is_data(phy_frame):
        return True
    if not 'MCPrimary1' in phy_frame.keys():
        get_weighted_primary(phy_frame, MCPrimary='MCPrimary1')
    return

def get_stream(phy_frame):
    if (phy_frame['I3EventHeader'].sub_event_stream == 'Final') | (phy_frame['I3EventHeader'].sub_event_stream=='InIceSplit'):
        return True
    else:
        return False

def run(i3_file, num_events, settings, geo_file, pulsemap_key,  do_classification=False):
    """IceTray script that wraps around an i3file and fills the events dict
       that is initialized outside the function

    Args:
        i3_file, and IceCube I3File
    Returns:
        True (IceTray standard)
    """

    # Initialize
    events = dict()
    events['reco_vals'] = []
    events['pulses'] = []
    events['waveforms'] = []
    events['pulses_timeseries'] = []
    events['t0'] = []
    surface = icecube.MuonGun.ExtrudedPolygon.from_file(geo_file, padding=0)
    def save_to_array(phy_frame):
        """Save the waveforms pulses and reco vals to lists.

        Args:
            phy_frame, and I3 Physics Frame
        Returns:
            True (IceTray standard)
        """
        reco_arr = []
        pulses = None
        for el in settings:
            if el[1] == pulsemap_key:
                try:
                    pulses = phy_frame[pulsemap_key].apply(phy_frame)
                except Exception as inst:
                    print('Failed to add pulses {} \n {}'.format(el[1], inst))
                    print inst
                    return False
            elif el[0] == 'variable':
                try:
                    reco_arr.append(eval('phy_frame{}'.format(el[1])))
                except Exception as inst:
#                    print inst
                    reco_arr.append(np.nan)
            elif el[0] == 'function':
                try:
                    reco_arr.append(
                        eval(el[1].replace('_icframe_', 'phy_frame, geometry_file')))
                except Exception as inst:
                    print('Failed to evaluate function {} \n {}'.format(el[1], inst))
                    return False
        tstr = 'Append Values for run_id {}, event_id {}'
        eheader = phy_frame['I3EventHeader']
        print(tstr.format(eheader.run_id, eheader.event_id))
        events['t0'].append(get_t0(phy_frame))
        events['pulses'].append(pulses)
        events['reco_vals'].append(reco_arr)
        return

    # I3Tray Defintion
    if isinstance(i3_file, list):
        files = [geo_file]
        files.extend(i3_file)
    else:
        files = [geo_file, i3_file]
    print files
    tray = I3Tray()
    tray.AddModule("I3Reader", "source",
                   FilenameList=files)
    tray.AddModule(get_stream, "get_stream",
                    Streams=[icetray.I3Frame.Physics])
#    tray.AddModule(print_short, 'info_short',
#                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(add_weighted_primary, "add_primary",
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(cuts2, 'cuts2',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.get_primary_nu, "primary_nu",
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.classify_wrapper, "classify",
                   surface=surface,
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.set_signature, "signature",
                   surface=surface,
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.first_interaction_point, "v_point",
                   surface=surface,
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.get_most_E_muon_info, 'energy info',
                   surface=surface,
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.track_length_in_detector, 'track_length',
                   surface=surface,
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.calc_depositedE, 'depo_energy',
                   surface=surface,
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(cuts, 'cuts',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.calc_hitDOMs, 'hitDOMs',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.get_inelasticity, 'inela',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(save_to_array, 'save',
                   Streams=[icetray.I3Frame.Physics])
    if num_events == -1:
        tray.Execute()
    else:
        tray.Execute(num_events)
    tray.Finish()

    return events
