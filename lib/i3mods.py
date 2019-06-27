from icecube import dataio, icetray, WaveCalibrator
from icecube import dataclasses, paraboloid, simclasses, recclasses, spline_reco
from I3Tray import *


def event_picker(phy_frame):
    # If we want to use that again rebuild that function
    if 'Scale_Class' in dataset_configparser.keys():
        for key in dataset_configparser['Scale_Class'].keys():
            scale_class[int(key)] = int(dataset_configparser['Scale_Class'][key])
    print('Scale classes like {}'.format(scale_class))
    if len(scale_class.keys()) > 0:
        max_scale = np.max([scale_class[key] for key in scale_class])
    else:
        max_scale = 1
    try:
        e_type = classify(phy_frame, geometry_file)
    except Exception as inst:
        print('The following event could not be classified')
        print(phy_frame['I3EventHeader'])
        print('First particle {}'.format(phy_frame['I3MCTree'][0].pdg_encoding))
        print(inst)
        return False
    rand = np.random.choice(range(1, max_scale+1))
    if e_type not in scale_class.keys():
        scaling = max_scale
    else:
        scaling = scale_class[e_type]
    if scaling >= rand:
        return True
    else:
        return False


def get_stream(phy_frame):
    if not phy_frame['I3EventHeader'].sub_event_stream == 'InIceSplit':
        return False
    else:
        return True


def save_to_array(phy_frame):
    """Save the waveforms pulses and reco vals to lists.

    Args:
        phy_frame, and I3 Physics Frame
    Returns:
        True (IceTray standard)
    """
    reco_arr = []
    if not z['ignore']:
        wf = None
    pulses = None
    if phy_frame is None:
        print('Physics Frame is None')
        return False
    for el in settings:
        if not z['ignore']:
            print z['ignore']
            if el[1] == '["CalibratedWaveforms"]':
                try:
                    wf = phy_frame["CalibratedWaveforms"]
                except Exception as inst:
                    print('uuupus {}'.format(el[1]))
                    print inst
                    return False
        elif el[1] == pulsemap_key:
            try:
                pulses = phy_frame[pulsemap_key].apply(phy_frame)
            except Exception as inst:
                print('Failed to add pulses {}'.format(el[1]))
                print inst
                print('Skip')
                return False
        elif el[0] == 'variable':
            try:
                reco_arr.append(eval('phy_frame{}'.format(el[1])))
            except Exception as inst:
                reco_arr.append(np.nan)
        elif el[0] == 'function':
            try:
                reco_arr.append(
                    eval(el[1].replace('_icframe_', 'phy_frame, geometry_file')))
            except Exception as inst:
                print('Failed to evaluate function {}'.format(el[1]))
                print(inst)
                print('Skip')
                return False

        # Removed part to append waveforms as it is depreciated
    if pulses is not None:
        tstr = 'Append Values for run_id {}, event_id {}'
        eheader = phy_frame['I3EventHeader']
        print(tstr.format(eheader.run_id, eheader.event_id))
        events['t0'].append(get_t0(phy_frame))
        events['pulses'].append(pulses)
        events['reco_vals'].append(reco_arr)
    else:
        print('No pulses in Frame...Skip')
        return False
    return
