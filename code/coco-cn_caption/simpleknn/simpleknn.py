
#!/usr/bin/env python

import time
from ctypes import *
from ctypes.util import find_library

import sys
import os

if "posix" == os.name:
    libsearch = CDLL(os.path.join(os.path.dirname(__file__), "cpp", "libsearch.so"))
else:
    filename = os.path.join(os.path.dirname(__file__), "cpp","windows","libsearch.dll")
    libsearch = cdll.LoadLibrary(filename)


DFUNC_MAPPING = {'l1':0, 'l2':1, 'chi2':2}

def fillprototype(f, restype, argtypes):
    f.restype = restype
    f.argtypes = argtypes

def genFields(names, types):
    return list(zip(names, types))


class search_result(Structure):
    _names = ["index", "value"]
    _types = [c_uint64, c_double]
    _fields_ = genFields(_names, _types)


class search_model(Structure):
    def __init__(self):
        self.__createfrom__ = 'python'

    def __del__(self):
        # free memory created by C to avoid memory leak
        if hasattr(self, '__createfrom__') and self.__createfrom__ == 'C':
            if pointer(self) is not None:
                libsearch.free_model(pointer(self))

    def load_ids(self, idfile):
        self.ids = str.split(open(idfile).readline().strip())
        assert(len(self.ids) == self.get_nr_images())

    def get_dim(self):
        return libsearch.get_dim(self)

    def get_nr_images(self):
        return libsearch.get_nr_images(self)

    def set_distance(self, dfunc):
        self.dfunc = DFUNC_MAPPING[dfunc]
        #print ('[%s] use %s distance' % (self.__class__.__name__, self.get_distance_name()))

    '''
    def useL2Distance(self):
        print ('[%s] use L2 distance' % self.__class__.__name__)
        self.l2 = 1
    '''

    def get_distance_name(self):
        NAMES = ['l1', 'l2', 'chi2']
        return NAMES[self.dfunc]

    def search_knn(self, query, max_hits):
        assert(len(query) == self.get_dim())
        topn = min(self.get_nr_images(), max_hits)

        query_ptr = (c_float * len(query))()
        for i in range(len(query)):
            query_ptr[i] = query[i]

        results = (search_result * topn)()
        s_time = time.time()
        libsearch.search_knn(self, query_ptr, topn, self.dfunc, results)
        knn_time = time.time() - s_time

        #print "search %d-nn, %.4f seconds" % (topn, knn_time)
        return [(self.ids[x.index], x.value) for x in results]


def toPyModel(model_ptr):
        """
        toPyModel(model_ptr) -> search_model

        Convert a ctypes POINTER(search_model) to a Python search_model
        """
        if bool(model_ptr) == False:
                raise ValueError("Null pointer")
        m = model_ptr.contents
        m.__createfrom__ = 'C'
        return m


def load_model(model_file_name, dim, nimages, id_file_name):
    model = libsearch.load_model(model_file_name, dim, nimages)
    if not model:
        print("failed to load a search model from %s" % model_file_name)
        return None

    model = toPyModel(model)
    model.load_ids(id_file_name)
    model.set_distance('l2')
    return model
    
    
fillprototype(libsearch.load_model, POINTER(search_model), [c_char_p, c_uint64, c_uint64])    
fillprototype(libsearch.get_dim, c_uint64, [POINTER(search_model)])
fillprototype(libsearch.get_nr_images, c_uint64, [POINTER(search_model)])
fillprototype(libsearch.free_model, None, [POINTER(POINTER(search_model))])
fillprototype(libsearch.search_knn, None, [POINTER(search_model), POINTER(c_float), c_uint64, c_int, POINTER(search_result)])

