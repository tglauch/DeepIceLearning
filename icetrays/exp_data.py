from icecube import dataio, icetray, WaveCalibrator
from icecube import dataclasses, paraboloid, simclasses, recclasses, spline_reco
from I3Tray import *
import sys
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.join(os.path.abspath(".."),'lib'))
import lib.reco_quantities as reco_q
from lib.functions_create_dataset import get_t0
import numpy as np
sys.path.append('/data/user/tglauch/DeepIceLearning/I3Module')
from i3module import DeepLearningClassifier

def print_info(phy_frame):
    print('run_id {} ev_id {}'.format(phy_frame['I3EventHeader'].run_id, phy_frame['I3EventHeader'].event_id))
    return

def cuts(phy_frame):
    """Performe a pre-selection of events according
       to the cuts defined in the config file

    Args:
        phy_frame, and IceCube I3File
    Returns:
        True (IceTray standard)
    """
    if (not phy_frame['QFilterMask']['CascadeFilter_12'].condition_passed) & \
       (not phy_frame['QFilterMask']['MuonFilter_12'].condition_passed):
        return False
    else:
        return True


def get_stream(phy_frame):
    if (phy_frame['I3EventHeader'].sub_event_stream == 'InIceSplit') & (phy_frame['I3EventHeader'].sub_event_id==0):
        return True
    else:
        return False

def run(i3_file, num_events, settings, geo_file, pulsemap_key, do_classification=False):
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
                    print inst
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

    if isinstance(i3_file, list):
        files = [geo_file]
        files.extend(i3_file)
    else:
        files = [geo_file, i3_file]
    tray = I3Tray()
    tray.AddModule("I3Reader", "source",
                   FilenameList=files)    
    tray.AddModule(get_stream, "get_stream",
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(cuts, 'cuts',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.calc_hitDOMs, 'hitDOMs',
                   Streams=[icetray.I3Frame.Physics])
 #   tray.AddModule(print_info, 'printer',
 #                  Streams=[icetray.I3Frame.Physics])
    if do_classification:
        tray.AddModule(DeepLearningClassifier, 'dl_class')
    tray.AddModule(save_to_array, 'save',
                   Streams=[icetray.I3Frame.Physics])
    if num_events == -1:
        tray.Execute()
    else:
        tray.Execute(num_events)
    tray.Finish()

    return events
