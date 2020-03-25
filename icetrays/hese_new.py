from icecube import dataio, icetray, phys_services
from icecube import dataclasses, paraboloid, simclasses, recclasses, spline_reco
from I3Tray import *
import sys
dil_path = '/data/user/tglauch/DeepIceLearning'
sys.path.append(dil_path)
sys.path.append(os.path.join(dil_path,'lib'))
import lib.reco_quantities as reco_q
from lib.functions_create_dataset import get_t0
import numpy as np
from icecube.weighting import weighting, get_weighted_primary
import icecube.MuonGun
sys.path.append('/home/tglauch/i3deepice/')
import argparse
from i3deepice.i3module import DeepLearningModule
from icecube import NewNuFlux


geo_file = '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2013.56429_V0.i3.gz'
surface = icecube.MuonGun.ExtrudedPolygon.from_file(geo_file, padding=0)

def get_primary(phy_frame):
    if reco_q.is_data(phy_frame):
        return True
    phy_frame.Put("MCTrack1", phy_frame['I3MCTree'][1])
    return


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

def cuts_muongun(phy_frame):
    if phy_frame['CausalQTot'] <6000:
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
    if phy_frame['QFilterMask']['SlopFilter_13'].condition_passed == 1:
        return False
    else:
        return True
    if phy_frame['HESE_CausalQTot'].value <0:
        return False
    else:
        return True

def get_stream(phy_frame):
    if (phy_frame['I3EventHeader'].sub_event_stream == 'best_fit') & (phy_frame['I3EventHeader'].sub_event_id==0):
        return True
    else:
        return False

def print_info(phy_frame):
    if reco_q.is_data(phy_frame):
        return True
    #print(phy_frame['MCPrimary1'].dir.zenith)
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



def atmo_weight(frame):
    if reco_q.is_data(frame):
        return True
    flux = NewNuFlux.makeFlux('honda2006')
    flux.knee_reweighting_model = "gaisserH4a_elbert"
    conv = frame['I3MCWeightDict']['OneWeight'] * flux.getFlux(frame['I3MCWeightDict']['PrimaryNeutrinoType'],
                                                               frame['MCPrimary1'].energy,
                                                               np.cos(frame['MCPrimary1'].dir.zenith))
    print('conv {}'.format(conv))
    frame.Put("conv", dataclasses.I3Double(conv))    
    return


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", help="files to be processed",
        type=str, nargs="+", required=True)
    parser.add_argument(
        "--pulsemap", type=str, default="SplitInIcePulses")
    parser.add_argument(
        "--batch_size", type=int, default=32)
    parser.add_argument(
        "--ncpus", type=int, default=1)
    parser.add_argument(
        "--ngpus", type=int, default=1)
    parser.add_argument(
        "--remove_daq", action='store_true', default=False)
    parser.add_argument(
        "--model", type=str, default='classification')
    parser.add_argument(
        "--dataset_config", type=str, default='None')
    parser.add_argument(
        "--replace", action='store_true',
        default=False)
    parser.add_argument(
        "--outfile", type=str, default='default')
    parser.add_argument(
        "--muongun", action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseArguments()
    print(args)
    files = []
    args.files =  np.atleast_1d(args.files)
    if (len(args.files)==1) & ('.txt' in args.files[0]):
         with open(args.files[0], 'r') as ifile:
            files= ifile.read().splitlines()
    else:
        for j in np.atleast_1d(args.files):
            if os.path.isdir(j):
                files.extend([os.path.join(j,i) for i in os.listdir(j) if '.i3' in i])
            else:
                files.append(j)
    files = sorted(files)
    if args.dataset_config is not 'None':
        from configparser import ConfigParser
        dataset_configparser = ConfigParser()
        dataset_configparser.optionxform = lambda option: option
        try:
            dataset_configparser.read(args.dataset_config)
            print "Config is found {}".format(args.dataset_config)
        except Exception as ex:
            raise Exception('Config File is missing or unreadable!!!!')
            print ex
        args.geo = str(dataset_configparser.get('Basics', 'geometry_file'))
        args.pulsemap = str(dataset_configparser.get('Basics', 'PulseSeriesMap'))
    i3_name = files[-1].split('/')[-1].split('.')[0]
    ofile = str(os.path.join(dataset_configparser.get('Basics', 'out_folder'),
                         i3_name + '_combined.i3.bz2'))
    print('Out File {}'.format(ofile))

    if args.muongun:
        generator = harvest_generators(list(files))
        model = icecube.MuonGun.load_model('GaisserH4a_atmod12_SIBYLL')

    files = np.concatenate([[args.geo],files])
    tray = I3Tray()
    tray.AddModule("I3Reader", "source",
                   FilenameList=files)
    if not args.muongun:
        tray.AddModule(cuts, "cuts",
                        Streams=[icetray.I3Frame.Physics])
    #    tray.AddModule(get_stream, "get_stream",
    #                    Streams=[icetray.I3Frame.Physics])
    tray.AddModule(print_short, 'info_short',
                   Streams=[icetray.I3Frame.Physics])
    if args.muongun:
        tray.AddModule(cuts_muongun, 'mg_cuts',
                      Streams=[icetray.I3Frame.Physics])
        tray.AddModule(get_primary, 'primary',
                       Streams=[icetray.I3Frame.Physics])
        tray.AddModule('I3MuonGun::WeightCalculatorModule', 'MuonWeight',
                        Model=model, Generator=generator)
        tray.AddModule(reco_q.classify_muongun, "classify",
                       surface=surface, primary_key = 'MCTrack1')
    if not args.muongun:
        tray.AddModule(add_weighted_primary, "add_primary",
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
#    tray.AddModule(reco_q.get_most_E_muon_info, 'energy info',
#                   surface=surface,
#                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.track_length_in_detector, 'track_length',
                   surface=surface,
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.calc_depositedE, 'depo_energy',
                   surface=surface,
                   Streams=[icetray.I3Frame.Physics])
    if not args.muongun:
        tray.AddModule(atmo_weight, 'conv_ow',
                       Streams=[icetray.I3Frame.Physics])
        tray.AddModule(reco_q.get_inelasticity, 'get_inelasticity',
                       Streams=[icetray.I3Frame.Physics])
        tray.AddModule(reco_q.coincidenceLabel_poly, 'coincidence',
                       Streams=[icetray.I3Frame.Physics])
    tray.AddModule(reco_q.calc_hitDOMs, 'hitDOMs',
                   pulsemap=args.pulsemap,
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule(DeepLearningModule, "DeepLearningMod",
                   pulsemap=args.pulsemap,
                   batch_size=args.batch_size,
                   save_as='TUM_classification',
                   cpu_cores=args.ncpus,
                   gpu_cores=args.ngpus,
                   remove_daq=args.remove_daq,
#                   calib_errata='CalibrationErrata',
#                   bad_dom_list='BadDomsList',
#                   saturation_windows='SaturationWindows',
#                   bright_doms='BrightDOMs',
                   model=args.model)
    tray.AddModule(print_info, 'pinfo',
                   Streams=[icetray.I3Frame.Physics])
    tray.Add("I3Writer",
             Filename=ofile,
             DropOrphanStreams=[icetray.I3Frame.Calibration,
                                icetray.I3Frame.DAQ])
    tray.Execute()
    tray.Finish()
    if True:
        for f in files:
            print('Remove {}'.format(f))
            os.remove(f)
