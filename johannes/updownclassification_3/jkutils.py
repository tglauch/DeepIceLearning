import os
import h5py
import numpy as np

# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
        
        
def read_files(input_files, data_location, using='time', virtual_len=-1, printfilesizes=False):
    input_data = []
    out_data = []
    file_len = []
    
  
    if printfilesizes:
        input_files = sorted(input_files)
        
    for run, input_file in enumerate(input_files):
        data_file = os.path.join(data_location, 'training_data/{}'.format(input_file))
  
        #print data_file, h5py.File(data_file).keys()
        if virtual_len == -1:
            data_len = len(h5py.File(data_file)[using])
        else:
            data_len = virtual_len
            print('Only use the first {} Monte Carlo Events'.format(data_len))
        if printfilesizes:
            print "{:10d}   {}".format(data_len, input_file)
        else:
            input_data.append(h5py.File(data_file, 'r')[using])
            out_data.append(h5py.File(data_file, 'r')['reco_vals'])
            file_len.append(data_len)
            #print type(input_data), type(input_data[-1]), type(input_data[-1][0]) #== list, h5py.Dataset, ndarray
            #print input_data[-1].shape #= (970452, 1, 21, 21, 51)
            
    return input_data, out_data, file_len


def zenith_to_binary(zenith):
    if type(zenith) == np.ndarray:
        ret = np.copy(zenith)
        ret[ret < 1.5707963268] = 0.0
        ret[ret > 1] = 1.0
        return ret
    if isinstance(zenith, float) or isinstance(zenith, int):
        return 1.0 if zenith > 1.5707963268 else 0.0
    if isinstance(zenith, list):
        temp_zenith = np.array(zenith)
        temp_zenith[temp_zenith < 1.5707963268] = 0.0
        temp_zenith[temp_zenith > 1] = 1.0
        return temp_zenith.tolist()
        """
        cur = zenith
        parent_cur = cur
        indize = []
        i = 0
        while type(cur) == list:
            parent_cur = cur
            indize.append(i)
            cur = cur[indize[-1]]
        for j in range(len(parent_cur)):
            exec("zenith{f} = 1 if zenith{f} > 1.5707963268 else 0".format(
                f="["+ "][".join(map(str,indize[:-1] + [j])) + "]"))
        """
        
def preprocess(times, replace_with=10):
    """
    print type(times) == np.ndarray
    print len(times) == 5000
    print type(times[0]) == np.ndarray
    print times[0].shape == (1,21,21,51)
    """
    ret = np.copy(times)
    ret[ret == np.inf] = replace_with
    return ret

def fake_preprocess(data, replace_with=0):
    return data