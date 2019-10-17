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
sys.path.append(os.environ['DNN_BASE'])
from i3deepice.i3module import DeepLearningClassifier
from icecube import DomTools, linefit
from icecube import gulliver,paraboloid,lilliput
from icecube import NewNuFlux
from icecube.icetray import I3Units

#from te_segment import Truncated
#from dedx_module import dEdx_fit

def cuts(phy_frame):
    """Performe a pre-selection of events according
       to the cuts defined in the config file

    Args:
        phy_frame, and IceCube I3File
    Returns:
        True (IceTray standard)
    """

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
    if not 'MCPrimary' in phy_frame.keys():
        get_weighted_primary(phy_frame)
    return

def get_stream(phy_frame):
    if reco_q.is_data(phy_frame):
        return True
    if (phy_frame['I3EventHeader'].sub_event_stream == 'Final') & (phy_frame['I3EventHeader'].sub_event_id==0):
        return True
    else:
        return False

soft = weighting.from_simprod(11029)
hard_lowE = weighting.from_simprod(11069)
hard_highE = weighting.from_simprod(11070)
generator = 3190 * soft + 3920 * hard_lowE + 997 * hard_highE
unit = I3Units.cm2/I3Units.m2

def calc_gen_ow(frame):
    if reco_q.is_data(frame):
        return True
    gen_w = generator(frame['MCPrimary1'].energy, frame['I3MCWeightDict']['PrimaryNeutrinoType'], np.cos(frame['MCPrimary1'].dir.zenith))
    pint = frame['I3MCWeightDict']['TotalWeight']
    ow = pint/gen_w/unit
    print('ow {}'.format(ow))
    frame.Put("correct_ow", dataclasses.I3Double(ow))
    return


def atmo_weight(frame):
    if reco_q.is_data(frame):
        return True
    flux = NewNuFlux.makeFlux('honda2006')
    flux.knee_reweighting_model = "gaisserH4a_elbert"
    conv = frame['correct_ow'].value * flux.getFlux(frame['I3MCWeightDict']['PrimaryNeutrinoType'],
                                                    frame['MCPrimary1'].energy,
                                                    np.cos(frame['MCPrimary1'].dir.zenith))
    print('conv {}'.format(conv))
    frame.Put("conv", dataclasses.I3Double(conv))    
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
    tray.AddModule(calc_gen_ow, 'gen_ow',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(atmo_weight, 'conv_ow',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.calc_hitDOMs, 'hitDOMs',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(cuts, 'cuts',
                   Streams=[icetray.I3Frame.Physics])
    if do_classification:
#        tray.AddModule(DeepLearningClassifier, 'dl_class',
#                       batch_size=8,
#                       n_cores=4,
#                       keep_daq=True,
#                       model='classification')
        tray.AddModule(DeepLearningClassifier, 'dl_energy',
                       batch_size=128,
                       n_cores=1,
                       model='mu_energy_reco_full_range',
                       save_as='dnn_reco')


    # Stuff for Hans
#    tray.AddService( "I3PhotonicsServiceFactory", "PhotonicsServiceMu_SpiceMie",
#        PhotonicsTopLevelDirectory="/cvmfs/icecube.opensciencegrid.org/data/photon-tables/SPICEMie/",
#        DriverFileDirectory="/cvmfs/icecube.opensciencegrid.org/data/photon-tables/SPICEMie/driverfiles",
#        PhotonicsLevel2DriverFile="mu_photorec.list",
#        PhotonicsTableSelection=2,
#        ServiceName="PhotonicsServiceMu_SpiceMie")

#    tray.AddSegment(Truncated,
#        Pulses="TWSRTHVInIcePulsesIC",
#        Seed="SplineMPE",
#        Suffix="",
#        PhotonicsService="PhotonicsServiceMu_SpiceMie",
#        Model="_SPICEMie")

#    tray.AddModule(dEdx_fit, 'dEdx_fit1', losses='newPS_SplineMPETruncatedEnergy_SPICEMie_BINS_dEdxVector')
#    tray.AddModule(dEdx_fit, 'dEdx_fit2', losses='SplineMPE_MillipedeHighEnergyMIE')
    tray.AddModule(reco_q.get_inelasticity, 'inela',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.millipede_max_loss, 'max_loss',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.millipede_std, 'millipede_std',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.millipede_n_losses, 'millipede_n_losses',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.millipede_rel_highest_loss, 'millipede_rel_highest_loss',
                   Streams=[icetray.I3Frame.Physics])
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
