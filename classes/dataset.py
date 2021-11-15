import csv
import glob
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt

''' the gan needs one batch of data at a time, nothing else. the data needs to be:
    - objects
    - minis
    - labels
      - (for CcGAN only)
      - we might also need a wholly separate list of labels to go along with the minis
    nothing else need be stored in memory at any given time, save perhaps for a list of the pathways to each
    file we'd like to include in the dataset (and perhaps a similar list for the label info, if we decide to 
    store it separately)
    when a call is made to the class from the gan, that's when the dataset class should batch data. 
    data must be saved according to the following format for objects, minis, and labels respectively:
    - objects:  shape = (B,1,2,c):      bsize, endres, dim2, channels (or (B,1,c) for time series) 
    - minis:    shape = (B,D,1,2,c):    bsize, ndownsamples, endres, dim2, channels ((B,D,1,c) for time series)
    - labels:   shape = (B,L):          bsize, nlabels
'''

class Dataset():

    def __init__(self, batch_size = 32, pathway = '', mdpath='', outpath = '', nlabels = 0, endres = 128):
        self.name           = 'dataset'
        self.batch_size     = batch_size
        self.batchlist      = []
        self.objectspath    = pathway       # the location of the directory to crawl for all usable objects
        self.outpath        = outpath       # place to save outbound images (will create dir if none exists)
        self.objectslist    = []            # list of pathways to all usable objects within objectspath and its subdirs
        self.nobjects       = 0
        self.endres         = endres          # dim2 is not explicitly stored, but should always be == dim2
        self.nchannels      = 2             # 3 for images, variable for time series
        self.nlabels        = nlabels       # num labels, categorical or numerical, to track (e.g. [m1, m2, EOS] == 3)
        self.labels         = {}
        self.minilabels     = {}
        self.objects        = []            # shape = (B,1,2,c): bsize, endres, endres, channels
        self.objtrans       = []            # self.objects, where each object is transposed
        self.minis          = []            # shape = (B,D,1,2,c): bsize, ndownsamples, endres, endres, channels
        self.outputs        = []            # images that will be processed as outbound images

        assert mdat != '', 'need to pass location of metadata file'
        assert pathway != '', 'need to pass location of folder that contains simulation files'
            self.compile_data(pathway, mdpath)


    # -----------------------------------------------------------------------------------------------
    def compile_data(self, pathway='', mdpath=''):
        print('Collecting objects for use...')
        self.objectspath = pathway
        self.objectslist = self.get_object_list_from_path()
        self.nobjects    = len(self.objectslist)
        # we want to save metadata file in memory here, as it will be needed frequently
        self.mdat        = pd.read_csv(mdpath, usecols=['lambda1','lambda2','m1','m2','file name'], sep='\t')
        print('Total objects found: ', self.nobjects)


    # -----------------------------------------------------------------------------------------------
    def get_object_list_from_path(self):
        objectslist     = glob.glob(self.objectspath + '/**/*.csv', recursive=True)
        return objectslist


    # -----------------------------------------------------------------------------------------------
    def create_batch(self):
        # get rid of old batch and make new one
        self.purge_batch()
        # currently just picking random objects, but we might want to do a queue instead for larger datasets
        self.batchlist = random.sample(self.objectslist, self.batch_size)
        for obj in self.batchlist:
            self.add_object_to_batch(obj)
        self.add_minis_to_batch()
        # self.add_labels_to_batch(???)


    # -----------------------------------------------------------------------------------------------
    def create_object(self, path):
        df    = pd.read_csv(path, header=None, usecols=[1,4], sep='\t')
        # df    = pd.read_csv(path, header=None, sep='\t')
        obj   = df.to_numpy()
        obj[:,0] = obj[:,0] * 4e21
        obj[:,1] = obj[:,1] * -1e-4
        for i in range(12):
            x = obj[::2**i]
            self.objects.append(x)


    # -----------------------------------------------------------------------------------------------
    def add_object_to_batch(self, path):
        df     = pd.read_csv(path, header=None, usecols=[1,4], sep='\t')
        x      = df.to_numpy()
        x[:,0] = x[:,0] * 4e21
        x[:,1] = x[:,1] * -1e-4
        obj = []
        for i in range(12):
            tmp = x[::2**i]
            obj.append(tmp)
        self.add_labels_to_object(obj, path)
        self.objects.append(obj) 


    # -----------------------------------------------------------------------------------------------
    def add_labels_to_object(self, obj, path):
        if self.nlabels != 0:
            # trying to get the file number out of file name '/../../../TSxxxxx.csv'
            name = path.split('TS')[-1].split('.')[0]
            # metadata's row # should be identical to file #
            row  = int(name) - 1
            lbls = mdat.iloc[row,:-1].to_numpy()
            # append array to obj
            obj.append(lbls)


    # -----------------------------------------------------------------------------------------------
    def save_objects(self, batch, location):
        i = 0
        for obj in batch:
            obj2 = np.array(obj)
            obj2[:,0] = obj2[:,0] / (4e21)
            obj2[:,1] = obj2[:,1] / (-1e-4)
            i += 1
            Path(location).mkdir(exist_ok=True)
            Path(location + '/series').mkdir(exist_ok=True)
            # naming convention can be thought out later
            np.savetxt(self.outpath + "/series/%02d.csv" % (i), obj2, delimiter='\t')


    # -----------------------------------------------------------------------------------------------
    def report_objects(self, batch, epoch, dim=0, cols=2, rows=10):
        pathway = 'drive/MyDrive/GWPAC/MSG-cGAN/Ours/ins/new_ts/sims/TS00001.csv'
        times   = []
        times   = pd.read_csv(pathway, header=None, usecols=[0], sep='\t')
        fig, axs = plt.subplots(rows, cols, figsize=(24,24))
        for i in range(10):
            axs[i, 0].plot(times, batch[i][:,0])
            axs[i, 1].plot(times, batch[i][:,1])
        Path(self.outpath).mkdir(exist_ok=True)
        plt.savefig(self.outpath + "/epoch%05d.png" % epoch)
        plt.close(fig)


    # -----------------------------------------------------------------------------------------------
    def purge_batch(self):
        self.labels         = []
        self.objects        = []
        self.batchlist      = []



# subclass for importing images as datasets
class Images(Dataset):

    def __init__(self, batch_size=32, pathway='', outpath='', nlabels=0, endres=128):
        super().__init__(batch_size=batch_size, pathway=pathway, outpath=outpath, nlabels=nlabels, endres=endres)
        self.name           = 'images'
        self.img            = [[]]
        self.nchannels      = 3


    # -----------------------------------------------------------------------------------------------
    def get_object_list_from_path(self):
        objectslist = glob.glob(self.objectspath + '/**/*.jpg', recursive=True)
        return objectslist


    # -----------------------------------------------------------------------------------------------
    def add_object_to_batch(self, path):
        x = Image.open(path)
        x.draft('RGB', (self.endres, self.endres))
        x = x.resize((self.endres, self.endres))
        x = np.array(x)
        x = (x - 127.5) / 127.5
        self.objects.append(x)


    # -----------------------------------------------------------------------------------------------
    # should create a list of dowsampled images, ranging from size (endres x endres) to (4 x 4)
    def add_minis_to_batch(self):
        res     = int(self.endres)
        l2      = int(math.log2(res))
        for i in range(l2 - 1):
            miniset = []
            for obj in self.batchlist:
                x = Image.open(obj)
                x = x.resize((res, res))
                x = np.array(x)
                x = (x - 127.5) / 127.5
                miniset.append(x)
            self.minis.append(miniset)
            res = int(res/2)    


    # -----------------------------------------------------------------------------------------------'
    def report_objects(self, batch, epoch, dim=0, cols=5, rows=5):
        if dim == 0:
            dim = self.endres
        image = np.zeros((dim * rows,  dim * cols, 3))
        for index, img in enumerate(batch):
            if index >= cols * rows:
                break
            h = dim
            w = dim
            i = index // cols
            j = index % cols
            image[i*h:(i+1)*h, j*w:(j+1)*w, :] = img[:, :, :]
        image = image*127.5 + 127.5
        image = Image.fromarray(image.astype(np.uint8))
        Path(self.outpath).mkdir(exist_ok=True)
        image.save(self.outpath + "/epoch%05d.png" % epoch)
