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

from i3module import DeepLearningClassifier
from icecube import NewNuFlux


def cuts(phy_frame):
    """Performe a pre-selection of events according
       to the cuts defined in the config file

    Args:
        phy_frame, and IceCube I3File
    Returns:
        True (IceTray standard)
    """

    return True

def harvest_generators(infiles):
    """
    Harvest serialized generator configurations from a set of I3 files.
    """
    import icecube
    import icecube.icetray
    from icecube import dataclasses, dataio, icetray
    import icecube.MuonGun
    from icecube.icetray.i3logging import log_info as log
    generator = None
    for fname in infiles:
        print fname
        f = dataio.I3File(str(fname))
        fr = f.pop_frame(icetray.I3Frame.Stream('S'))
        f.close()
        if fr is not None:
            for k in fr.keys():
                v = fr[k]
                if isinstance(v, icecube.MuonGun.GenerationProbability):
    #                log('%s: found "%s" (%s)' % (fname, k, type(v).__name__), unit="MuonGun")
                    if generator is None:
                        generator = v
                    else:
                        generator += v
    return generator


def get_stream(phy_frame):
    if (phy_frame['CausalQTot'] > 0):
        return True
    else:
        return False

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

def get_primary(phy_frame):
    if reco_q.is_data(phy_frame):
        return True
    phy_frame.Put("MCTrack1", phy_frame['I3MCTree'][1])
    return


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
        events['t0'].append(get_t0(phy_frame, puls_key=pulsemap_key))
        events['pulses'].append(pulses)
        events['reco_vals'].append(reco_arr)
        return

    # I3Tray Defintion
    generator = harvest_generators(list(i3_file))
    if isinstance(i3_file, list):
        files = [geo_file]
        files.extend(i3_file)
    else:
        files = [geo_file, i3_file]
    print files
    model = icecube.MuonGun.load_model('GaisserH4a_atmod12_SIBYLL')
    tray = I3Tray()
    tray.AddModule("I3Reader", "source",
                   FilenameList=files)
    tray.AddModule(get_stream, "get_stream",
                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(print_short, 'info_short',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(get_primary, 'primary',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule('I3MuonGun::WeightCalculatorModule', 'MuonWeight',
                    Model=model, Generator=generator)
    tray.AddModule(reco_q.classify_muongun, "classify",
                   surface=surface, primary_key = 'MCTrack1', 
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
    tray.AddModule(reco_q.calc_hitDOMs, 'hitDOMs',
                   Streams=[icetray.I3Frame.Physics],
                   pulsemap=pulsemap_key)
    tray.AddModule(cuts, 'cuts',
                   Streams=[icetray.I3Frame.Physics])
    if do_classification:
        tray.AddModule(DeepLearningClassifier, 'dl_class', pulsemap=pulsemap_key)
    tray.AddModule(print_info, 'pinfo',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(save_to_array, 'save',
                   Streams=[icetray.I3Frame.Physics])
    if num_events == -1:
        tray.Execute()
    else:
        tray.Execute(num_events)
    tray.Finish()

    return events
