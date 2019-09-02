import os
import numpy as np
import scandir

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

def keep_valid(list1, remove_keys):
    nlist = []
    for l in list1:
        valids = [i in l for i in remove_keys]
        if np.any(valids):
            nlist.append(l)
        else:
            continue
    return nlist

def remove_invalid(list1, remove_keys):
    nlist = []
    for l in list1:
        valids = [i in l for i in remove_keys]
        if np.any(valids):
            print l
            continue
        else:
            nlist.append(l)
    return nlist 


def get_files_from_folder(basepath, folderlist, compression_format, filelist, must_contain, exclude):       
    if folderlist == 'allinmcpath':
        folderlists = []
        for p in basepath:
            tlist = []
            for root, dirs, files in scandir.walk(p):
                print root
                if root == 'logs':
                    print('Skip Logs')
                    continue
                a =  [s_file for s_file in files
                     if (s_file[-6:] in compression_format) &
                        (not '_SLOP.' in s_file) & (not '_IT.' in s_file)&
                        (not 'GCD.' in s_file) & (not 'EHE.' in s_file)]
                if must_contain is not None:
                    a = keep_valid(a, must_contain)
                if exclude is not None:
                    a = remove_invalid(a, exclude)
                if len(a) > 0:
                    tlist.append(root)
            folderlists.append(tlist)
    else:
         folderlists = [[folder.strip() for folder in folderlist.split(',')]]

    if not filelist == 'allinfolder':
        filelist = filelist.split(',')
    num_files = []
    run_filelist = []
    for j, bfolder in enumerate(folderlists):

        run_filelist.append([])
        for subpath in bfolder:
            i3_files_all = [os.path.join(subpath, s_file) for s_file in os.listdir(subpath)
                            if (s_file[-6:] in compression_format) & (not os.path.isdir(s_file)) & 
                            (not '_SLOP.' in s_file) & (not '_IT.' in s_file) &
                            (not 'GCD.' in s_file) & (not 'EHE.' in s_file)]
            if must_contain is not None:
                i3_files_all = keep_valid(i3_files_all, must_contain)
            if exclude is not None:
                i3_files_all = remove_invalid(i3_files_all, exclude)
            if not filelist == 'allinfolder':
                i3_files = [f for f in filelist if f in i3_files_all]
            else:
                i3_files = i3_files_all
            print('Number of I3Files found {}'.format(len(i3_files)))
            run_filelist[j].extend(i3_files)
    return run_filelist, num_files


def split_filelist():
    return
