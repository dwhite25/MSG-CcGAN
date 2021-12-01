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

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
class Dataset():    
    # -----------------------------------------------------------------------------------------------
    def __init__(self, batch_size = 32, pathway = '', mdpath='', outpath = '', nlabels = 0, endres = 128):
        self.name           = 'dataset'
        self.batch_size     = batch_size
        self.batchlist      = []
        self.objectspath    = pathway       # the location of the directory to crawl for all usable objects
        self.outpath        = outpath       # place to save outbound images (will create dir if none exists)
        self.objectslist    = []            # list of pathways to all usable objects within objectspath and its subdirs
        self.nobjects       = 0
        self.endres         = endres        # == the highest resolution of the inboun/outbound time series data
        self.nchannels      = 2             # for time series, current default is 2 (amplitude and phase)
        self.nlabels        = nlabels       # num labels, categorical or numerical, to track (e.g. [m1, m2, l1, l2] == 4)
        self.labels         = []            # the labels for a single simulation in the case we create one at a time (e.g. "'makewaves')
        self.labels_batch   = []            # batch of labels that corresponds 1:1 to batch of time series sims fed into the network
        self.objects        = []            # shape = (B,1,c): bsize, endres, channels
        self.outputs        = []            # images that will be processed as outbound images
        self.mdat           = []            # eventual holder for all metadata info, as pulled from the .csv file

        assert mdpath != '', 'need to pass location of metadata file'
        assert pathway != '', 'need to pass location of folder that contains simulation files'

        self.compile_data(pathway, mdpath)

    # -----------------------------------------------------------------------------------------------
    def compile_data(self, pathway='', mdpath=''):
        print('Collecting objects for use...')
        self.objectspath = pathway
        self.objectslist = self.get_object_list_from_path()
        self.nobjects    = len(self.objectslist)
        # we want to save metadata file in memory for the entire run here, as it will be needed frequently
        self.mdat        = pd.read_csv(mdpath, usecols=['lambda1','lambda2','m1','m2','file name'], sep='\t')
        print('Total objects found: ', self.nobjects)

    # -----------------------------------------------------------------------------------------------
    def get_object_list_from_path(self):
        objectslist     = glob.glob(self.objectspath + '/**/*.csv', recursive=True)

        return objectslist

    # -----------------------------------------------------------------------------------------------
    # creates batch of self.batch_size time series objects and batch of self.batch-size corresponding
    def create_batch(self):
        # get rid of old batch and make new one
        self.purge_batch()
        # currently just picking the object randomly from the list, rather than using a queue
        self.batchlist = random.sample(self.objectslist, self.batch_size)
        for path in self.batchlist:
            self.add_object_to_batch(path)
            self.add_labels_to_batch(path)
    
    # -----------------------------------------------------------------------------------------------
    # final object array should be [[full res],[downsample_1],[downsample_2],...,[downsample_n]]
    # where each downsample is 1/2 the resolution of the previous.
    # also saves coresponding labels in memory (optional)
    def create_object(self, path, lbls=False):
        x      = pd.read_csv(path, header=None, usecols=[1,4], sep='\t')
        x      = x.to_numpy(dtype=np.float32)
        # keep values within (0,1) and not too close to extremes in general. 
        # (this means all those zeroes in amplitude need to be offset)
        x[:,0] = x[:,0] * 3e21 + 0.20
        x[:,1] = x[:,1] * -1e-4
        obj    = []
        for i in range(12):
            obj.append(np.array(x[::2**i], dtype=np.float32))
        if lbls == True:
            self.labels = create_labels(obj, path)
        obj = np.array(obj)

        return obj

    # -----------------------------------------------------------------------------------------------
    # adds one time series object to batch
    def add_object_to_batch(self, path):
        obj = self.create_object(path)
        self.objects.append(obj)

    # -----------------------------------------------------------------------------------------------
    # similar to create_object, but for labels. can work with either create_object or add_labels_to_batch
    def create_labels(self, path):
        # get the file number out of file name '/../../../TSxxxxx.csv'
        name = path.split('TS')[-1].split('.')[0]
        # metadata's row # should be one off from file #
        row  = int(name) - 1
        lbls = self.mdat.iloc[row,:-1].to_numpy(dtype=np.float32)
        # important to note that labels come out as [l1, l2, m1, m2] in that order.

        return lbls

    # -----------------------------------------------------------------------------------------------
    # adds one set of labels, corresponding with one time series object, to the batch of labels
    def add_labels_to_batch(self, path):
        lbls = self.create_labels(path)
        self.labels_batch.append(lbls)

    # -----------------------------------------------------------------------------------------------
    # saves copies of time series data coming out of the GAN as a .csv file.
    # might want to think of a naming convention that embeds info about masses and lambdas
    def save_objects(self, batch, location):
        i = 0
        for obj in batch:
            obj2 = np.array(obj)
            obj2[:,0] = (obj2[:,0] - 0.20) / (3e21)
            obj2[:,1] = obj2[:,1] / (-1e-4)
            i += 1
            Path(location).mkdir(exist_ok=True)
            Path(location + '/series').mkdir(exist_ok=True)
            # naming convention can be thought out later
            np.savetxt(self.outpath + "/series/%02d.csv" % (i), obj2, delimiter='\t')

    # -----------------------------------------------------------------------------------------------
    # takes the timestamps from a random sim, since timestamps weren't used in the NN
    # then creates an image stack of the amplitude and strain of a sim created in the NN
    def report_objects(self, batch, epoch, dim=0, cols=2, rows=10, normalized=True):
        pathway = self.objectspath + '/1600/40/TS00001.csv'
        times   = []
        times   = pd.read_csv(pathway, header=None, usecols=[0], sep='\t')
        fig, axs = plt.subplots(rows, cols, figsize=(24,24))
        batch2  = []
        if normalized == False:
            for obj in batch:
                obj2 = np.array(obj)
                obj2[:,0] = (obj2[:,0] - 0.20) / (3e21)
                obj2[:,1] = obj2[:,1] / (-1e-4)
                batch2.append(obj2)
            for i in range(10):
                axs[i, 0].plot(times, batch2[i][:,0])
                axs[i, 1].plot(times, batch2[i][:,1])
        else:
            for i in range(10):
                axs[i, 0].plot(times, batch[i][:,0])
                axs[i, 1].plot(times, batch[i][:,1])
        Path(self.outpath).mkdir(exist_ok=True)
        plt.savefig(self.outpath + "/epoch%05d.png" % epoch)
        plt.close(fig)

    # -----------------------------------------------------------------------------------------------
    # empty out all data to get ready for new batch
    def purge_batch(self):
        self.labels         = []
        self.objects        = []
        self.batchlist      = []
        self.labels_batch   = []
