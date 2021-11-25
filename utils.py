# data storage
import _pickle as cpkl
import bz2

# Pickle a file and then compress it into a file with extension 
"""
usage: 
    compressed_pickle('path/filename', data) 
"""

def compressed_cpickle(path_title, data):
    with bz2.BZ2File(path_title+'.pbz2', "w") as f: 
        cpkl.dump(data, f)

        
# Load any compressed pickle file
"""
usage: 
    data = decompress_pickle('example_cp.pbz2') 
"""
def decompress_cpickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cpkl.load(data)
    return data