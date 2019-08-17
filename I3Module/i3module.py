import sys
import os
dirname = os.path.dirname(__file__)
sys.path.insert(0, '/home/tglauch/virtualenvs/tf_env3/lib/python2.7/site-packages')
sys.path.insert(0, os.path.join(dirname, 'lib/'))
import tensorflow as tf
from model_parser import parse_functional_model
from helpers import *
import numpy as np
from icecube import icetray
from I3Tray import I3Tray
from configparser import ConfigParser
from collections import OrderedDict
import time
from icecube.dataclasses import I3MapStringDouble
from icecube import dataclasses, dataio
import argparse

class DeepLearningClassifier(icetray.I3ConditionalModule):
    def __init__(self,context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("pulsemap","Define the name of the pulsemap",
                          "InIceDSTPulses")
        self.AddParameter("save_as", "Define the Output key",
                          "Deep_Learning_Classification")
        print('Init Deep Learning Classifier..this may take a while')

    def Configure(self):
        self._runinfo = np.load(os.path.join(dirname,'cfg/run_info.npy'), allow_pickle=True)[()]
        self._grid = np.load(os.path.join(dirname, 'cfg/grid.npy'), allow_pickle=True)[()]
        self._inp_shapes = self._runinfo['inp_shapes']
        self._out_shapes = self._runinfo['out_shapes']
        self._inp_trans = self._runinfo['inp_trans']
        self._out_trans = self._runinfo['out_trans']
        self._pulsemap = self.GetParameter("pulsemap") 
        self._save_as =  self.GetParameter("save_as")
        print("Im configured with {} and {}".format(self._pulsemap,self._save_as))
        import cfg.model as func_model_def
        self._model = func_model_def.model(self._inp_shapes, self._out_shapes)
        self._model.load_weights(os.path.join(dirname, 'cfg/weights.npy'))
        dataset_configparser = ConfigParser()
        dataset_configparser.read(os.path.join(dirname,'cfg/config.cfg'))
        inp_defs = dict()
        for key in dataset_configparser['Input_Times']:
            inp_defs[key] = dataset_configparser['Input_Times'][key]
        for key in dataset_configparser['Input_Charges']:
            inp_defs[key] = dataset_configparser['Input_Charges'][key]
        self._inputs = []
        for key in self._inp_shapes.keys():
            binput = []
            branch = self._inp_shapes[key]
            for bkey in branch.keys():
                if bkey == 'general':
                    continue
                elif 'charge_quantile' in bkey:
                    feature = 'pulses_quantiles(charges, times, {})'.format(float('0.' + bkey.split('_')[3]))
                else:
                    feature = inp_defs[bkey.replace('IC_','')]
                trans = self._inp_trans[key][bkey]
                binput.append((feature, trans))
            self._inputs.append(binput)


    def Physics(self, frame):
        timer_t0 = time.time()
        #This runs on P-frames
        pulse_key = self._pulsemap
        f_slice = []
        t0 = get_t0(frame, puls_key=pulse_key)
        pulses = frame[pulse_key].apply(frame)
        for key in self._inp_shapes.keys():
            f_slice.append(np.zeros(self._inp_shapes[key]['general']))
        for omkey in pulses.keys():
            dom = (omkey.string, omkey.om)
            if not dom in self._grid.keys():
                continue
            gpos = self._grid[dom]
            charges = np.array([p.charge for p in pulses[omkey][:]])
            times = np.array([p.time for p in pulses[omkey][:]]) - t0
            widths = np.array([p.width for p in pulses[omkey][:]])
            for branch_c, inp_branch in enumerate(self._inputs):
                for inp_c, inp in enumerate(inp_branch):
                    f_slice[branch_c][gpos[0]][gpos[1]][gpos[2]][inp_c] = inp[1](eval(inp[0]))
        processing_time = time.time() - timer_t0
        prediction = self._model.predict(np.array(f_slice, ndmin=5), batch_size=None, verbose=0,
                                   steps=None)
        prediction_time = time.time() - processing_time - timer_t0
        output = I3MapStringDouble()
        output['Skimming'] = float(prediction[0][0])
        output['Cascade'] = float(prediction[0][1])
        output['Through_Going_Track'] = float(prediction[0][2])
        output['Starting_Track'] = float(prediction[0][3])
        output['Stopping_Track'] = float(prediction[0][4])
        frame.Put(self._save_as, output)
        #print('Total Time {:.2f} s, Processing Time {:.2f}s, Prediction Time {:.2f}s'.format(time.time() - timer_t0, processing_time, prediction_time))
        self.PushFrame(frame)

    def DAQ(self,frame):
        #This runs on Q-Frames
        self.PushFrame(frame)

    def Finish(self):
        #Here we can perform cleanup work (closing file handles etc.)
        pass

def print_info(phy_frame):
    print('run_id {} ev_id {}'.format(phy_frame['I3EventHeader'].run_id, phy_frame['I3EventHeader'].event_id))
    print(phy_frame["Deep_Learning_Classification"])
    return

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np

def figsize(scale, ratio=(np.sqrt(5.0)-1.0)/2.0):
    fig_width_pt = 455.8843                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*ratio            # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size


def plot_prediction(prediction, figax=(None,None)):
    data = np.arange(len(prediction))
    bins = np.arange(0, data.max() + 1.5) - 0.5
    fig = plt.figure(figsize=figsize(0.9, (np.sqrt(5.0)-1.0)/2.0))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.barh(data, width=prediction, height=0.2,
    tick_label=['Non-Starting Cascade', 'Starting Cascade', 'Through-Going Track',
                'Starting Track', 'Stopping Track'], alpha=0.9)
    ax.set_xlabel('Prediction Score')
    return fig, ax

def make_plot(frame, key="Deep_Learning_Classification"):
    ofolder = os.path.join(dirname, 'plots')
    if not os.path.exists(ofolder):
        os.makedirs(ofolder)

    prediction = [frame[key]['Skimming'],
                  frame[key]['Cascade'],
                  frame[key]['Through_Going_Track'],
                  frame[key]['Starting_Track'],
                  frame[key]['Stopping_Track']]
    prediction = np.array(prediction)
    fig, ax = plot_prediction(prediction)
    fig.savefig(os.path.join(ofolder, '{}_{}.pdf'.format(frame['I3EventHeader'].run_id, frame['I3EventHeader'].event_id)))
    return


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        help="files to be processed",
        type=str, nargs="+", required=True)
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False)
    parser.add_argument(
        "--pulsemap",
        type=str, default="InIceDSTPulses")    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArguments()
    files = []
    for j in np.atleast_1d(args.files):
        if os.path.isdir(j):
            files.extend([os.path.join(j,i) for i in os.listdir(j) if '.i3' in i])
        else:
            files.append(j)
    files = sorted(files)
    tray = I3Tray()
    tray.AddModule('I3Reader','reader',
                   FilenameList = files)
    tray.AddModule(DeepLearningClassifier, "DeepLearningClassifier", pulsemap=args.pulsemap)
    tray.AddModule(print_info, 'printer',
                   Streams=[icetray.I3Frame.Physics])
    if args.plot:
        tray.AddModule(make_plot, 'plotter',
                       Streams=[icetray.I3Frame.Physics])
    tray.Execute()
    tray.Finish()
