#!/usr/bin/env python
# coding: utf-8

"""This file is part of DeepIceLearning
DeepIceLearning is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import tables
import numpy as np
import importlib
import os
import sys
from collections import OrderedDict


def prepare_io_shapes(inputs, outputs, exp_file):
    inp_transformations = OrderedDict()
    inp_shapes = OrderedDict()
    out_transformations = OrderedDict()
    out_shapes = OrderedDict()
    # open example file
    inp_file = tables.open_file(exp_file)

    for br in inputs:
        inp_shapes[br] = {}
        inp_transformations[br] = {}
        for var, tr in zip(inputs[br]["variables"],
                           inputs[br]["transformations"]):
            test_arr = np.array(inp_file._get_node("/" + var)[0])
            res_shape = np.shape(np.squeeze(tr(test_arr))) if not \
                    isinstance(tr(test_arr), np.float) else (1,)
            print(br,var,res_shape)
            inp_shapes[br][var] = res_shape
            inp_transformations[br][var] = tr
        if len(inputs[br]["variables"]) > 1:
            inp_shapes[br]["general"] = \
                    res_shape + (len(inputs[br]["variables"]),)
        else:
            if len(res_shape) >1:
                inp_shapes[br]["general"] = res_shape + (1,)
            else:
                inp_shapes[br]["general"] = res_shape

    for br in outputs:
        out_shapes[br] = {}
        out_transformations[br] = {}
        for var, tr in zip(outputs[br]["variables"],
                           outputs[br]["transformations"]):
            test_arr = np.array(inp_file._get_node("/reco_vals").col(var)[0])
            res_shape = np.shape(np.squeeze(tr(test_arr))) if not \
                    isinstance(tr(test_arr), np.float) else (1,)
            print(br,var,res_shape)
            out_shapes[br][var] = res_shape
            out_transformations[br][var] = tr
        if len(outputs[br]["variables"]) > 1:
            out_shapes[br]["general"] = \
                res_shape + (len(outputs[br]["variables"]),)
        else:
            if len(res_shape) > 1:
                out_shapes[br]["general"] = res_shape + (1,)
            else:
                out_shapes[br]["general"] = res_shape

    return inp_shapes, inp_transformations, out_shapes, out_transformations

def parse_reference_output(cfg_file):
    try:
        sys.path.append(os.path.dirname(cfg_file))
        mname = os.path.splitext(os.path.basename(cfg_file))[0]
        func_model_def = importlib.import_module(mname)
        sys.path.pop()
    except Exception:
        raise Exception('Import of model.py failed: {}'.format(cfg_file))
    ref_outputs = func_model_def.reference_outputs
    return ref_outputs

def parse_functional_model(cfg_file, exp_file):
    # fancy relative imports..
    sys.path.append(os.path.dirname(cfg_file))
    #sys.path.append("/scratch9/mkron/software/DeepIceLearning/Networks/classifikation_mk/")
    #sys.path.append("/scratch9/mkron/data/NN_out/run35/")
    sys.path.append(os.getcwd()+"/"+os.path.dirname(cfg_file))
    mname = os.path.splitext(os.path.basename(cfg_file))[0]
    func_model_def = importlib.import_module(mname)
    sys.path.pop()
    # except Exception:
    #    raise Exception('Import of model.py failed: {}'.format(cfg_file))
    inputs = func_model_def.inputs
    outputs = func_model_def.outputs

    in_shapes, in_trans, out_shapes, out_trans = \
        prepare_io_shapes(inputs, outputs, exp_file)
    print(out_shapes)
    print(in_shapes)
    model = func_model_def.model(in_shapes, out_shapes)
    return model, in_shapes, in_trans, out_shapes, out_trans


