# -*- coding: future_fstrings -*-

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-22

# System packages.
import json
import os

# PyTorch packages.
import torch.utils.data
import torchvision.transforms as transforms

# Local packages.
from . import preprocessor
from .file_list_loader import FileListLoader
from . import assemble_list_loader
from .readers import READERS
from .utils import read_string_list_2D

from .register import ( PRE_PROCESSORS, DATASET_LOADERS, make_object )

DT_JSON_NAME        = 'name'
DT_JSON_DT_ROOT     = "datasetRoot"
DT_JSON_FL_DIR      = "fileListDir"
DT_JSON_TRAIN       = "trainFileList"
DT_JSON_TRAIN_STAT_MEAN = "trainStatMean"
DT_JSON_TEST        = "testFileList"
DT_JSON_ITT         = "flagTestInTraining"
DT_JSON_INFER       = "inferFileList"
DT_JSON_OCC_TRAIN   = "trainOccFileList"
DT_JSON_OCC_TEST    = "testOccFileList"
DT_JSON_USE_OCC     = "flagUseOcc"
DT_JSON_VALID_TRAIN = 'trainValidMaskList'
DT_JSON_VALID_TEST  = 'testValidMaskList'
DT_JSON_DLMT        = "delimiter"
DT_JSON_F_DEPTH     = "flagDepth"
DT_JSON_FB          = "fb"
DT_JSON_DISP_READER = 'dReader'
DT_JSON_OCC_READER  = 'occReader'

def read_json(fn):
    if ( not os.path.isfile(fn) ):
        raise Exception("{} does not exist. ".format(fn))

    with open(fn, "r") as fp:
        j = json.load(fp)

    return j

def create_data_representation_from_json_obj(jObj):
    cols = read_string_list_2D( 
        os.path.join( jObj[DT_JSON_FL_DIR], jObj[DT_JSON_TEST] ), 
        3, delimiter=jObj[DT_JSON_DLMT] )

    occFileListFn = jObj[DT_JSON_OCC_TEST]
    if ( occFileListFn != '' ):
        occ = read_string_list_2D( 
            os.path.join(jObj[DT_JSON_FL_DIR], occFileListFn), 
            1, delimiter=jObj[DT_JSON_DLMT] )[0]
    else:
        occ = None

    return assemble_list_loader.DatasetRepresentation( 
        jObj[DT_JSON_NAME], jObj[DT_JSON_DT_ROOT], 
        cols[0], cols[1], cols[2], 
        filesOcc=occ, 
        fb=jObj[DT_JSON_FB], 
        dReader=jObj[DT_JSON_DISP_READER], 
        occReader=jObj[DT_JSON_OCC_READER], 
        flagUseOcc=jObj[DT_JSON_USE_OCC] )

class CompoundLoader(object):
    def __init__(self,
        datasetJSONList,
        batchSize=2, shuffle=True, numWorkers=2, dropLast=False,
        testBatchFactor=1.0,
        loaderConfMain=None,
        loaderConfITT=None ):
        super(CompoundLoader, self).__init__()

        self.flagTest = False

        # datasetJSONList is a list of dicts.
        # Every element has a structure of { 'description':, 'repeat': }
        self.datasetJSONList  = datasetJSONList

        self.batchSize        = batchSize
        self.shuffle          = shuffle
        self.numWorkers       = numWorkers
        self.dropLast         = dropLast

        # Use values smaller than 1.0 to assign a 
        # # smaller batch for non-ITT test during training.
        assert( 0 < testBatchFactor <= 1.0 )
        self.testBatchFactor = testBatchFactor
        
        self.loaderTrain  = None
        self.loaderTest   = None

        self.dataITT   = None # ITT stands for in-training test. A data.Dataset object.
        self.loaderITT = None

        self.loaderConfMain = loaderConfMain \
            if loaderConfMain is not None \
            else dict( 
                loaderTrain=FileListLoader.get_default_init_args(training=True),
                loaderTest=FileListLoader.get_default_init_args(training=False) )

        self.loaderConfITT = loaderConfITT \
            if loaderConfITT is not None \
            else assemble_list_loader.PreloadedAssemble.get_default_init_args()

    def set_test(self):
        self.flagTest = True

    def create_datasets_from_json_list(self):
        statMeanCondition = 0 # 0 is initial condition, 1 means valid condition, 2 means invalid condition.

        trainDatasets = []
        testDatasets  = []

        # import ipdb; ipdb.set_trace()
        
        for datasetDict in self.datasetJSONList:
            fn = datasetDict['description']
            repeat = datasetDict['repeat']

            if ( not os.path.isfile(fn) ):
                raise Exception("{} not exist. ".format(fn))

            try:
                # Read the json file.
                jObj = read_json(fn)

                # Get the file lists for a training dataset.
                if ( DT_JSON_TRAIN in jObj ):
                    imgL, imgR, disp = read_string_list_2D( \
                        jObj[DT_JSON_FL_DIR] + "/" + jObj[DT_JSON_TRAIN], \
                        3, delimiter=jObj[DT_JSON_DLMT], prefix=jObj[DT_JSON_DT_ROOT] )

                    # Check if we have the stat info.
                    if ( DT_JSON_TRAIN_STAT_MEAN in jObj ):
                        if ( statMeanCondition == 2 ):
                            raise Exception('Current dataset {} contains stat info while there is at least one dataset that does not provide stat info. '\
                                .format(fn))
                        statMean = np.load( 
                            os.path.join(jObj[DT_JSON_FL_DIR], jObj[DT_JSON_TRAIN_STAT_MEAN]) ).reshape((-1, )).astype(np.float32)

                        statMeanCondition = 1
                    else:
                        if ( statMeanCondition == 1 ):
                            raise Exception('Current dataset {} does not contain stat info while there is at least one dataset that contains stat info. '.\
                                format(fn))
                        statMeanCondition = 2

                    if ( jObj[DT_JSON_F_DEPTH] ):
                        fb = jObj[DT_JSON_FB]

                        # Although the input is depth, but statMean is always in disparity.
                    else:
                        fb = -1.0

                    # The loaders.
                    imgLoader  = READERS['ImgPlain']()
                    dispLoader = READERS[ jObj[DT_JSON_DISP_READER] ](fb)
                    occLoader  = READERS[ jObj[DT_JSON_OCC_READER] ]()

                    # Create a training dataset.
                    datasetTrain = make_object(
                        DATASET_LOADERS, self.loaderConfMain['loaderTrain'])
                    datasetTrain.set_lists( 
                        left=imgL, right=imgR, dispL=disp)
                    datasetTrain.set_readers(
                        imgReader=imgLoader, dispReader=dispLoader,
                        occReader=occLoader )

                    if ( DT_JSON_OCC_TRAIN in jObj and jObj[DT_JSON_OCC_TRAIN] != '' ):
                        [occL] = read_string_list_2D( \
                            jObj[DT_JSON_FL_DIR] + "/" + jObj[DT_JSON_OCC_TRAIN], \
                            1, delimiter=jObj[DT_JSON_DLMT], prefix=jObj[DT_JSON_DT_ROOT] )

                        assert( len(occL) == len(imgL) )
                        datasetTrain.set_occ_L( occL )

                    if ( DT_JSON_VALID_TRAIN in jObj and jObj[DT_JSON_VALID_TRAIN] != '' ):
                        [validMask] = read_string_list_2D(
                            jObj[DT_JSON_FL_DIR] + "/" + jObj[DT_JSON_VALID_TRAIN],
                            1, delimiter=jObj[DT_JSON_DLMT], prefix=jObj[DT_JSON_DT_ROOT] )

                        assert( len(validMask) == len(imgL) )
                        datasetTrain.set_valid_mask(validMask)

                    # # Flag for occ.
                    # if ( jObj[DT_JSON_USE_OCC] ):
                    #     datasetTrain.flagUseOcc = True
                    # else:
                    #     datasetTrain.flagUseOcc = False

                    trainDatasets = trainDatasets + [ datasetTrain ] * repeat

                # Get the file lists for a testing dataset.
                if ( not DT_JSON_ITT in jObj or not jObj[DT_JSON_ITT] ):
                    if ( DT_JSON_TEST in jObj ):
                        imgL, imgR, disp = read_string_list_2D( \
                            jObj[DT_JSON_FL_DIR] + "/" + jObj[DT_JSON_TEST], \
                            3, delimiter=jObj[DT_JSON_DLMT], prefix=jObj[DT_JSON_DT_ROOT] )

                        if ( jObj[DT_JSON_F_DEPTH] ):
                            fb = jObj[DT_JSON_FB]
                        else:
                            fb = -1.0

                        # The loaders.
                        imgLoader  = READERS['ImgPlain']()
                        dispLoader = READERS[ jObj[DT_JSON_DISP_READER] ](fb)
                        occLoader  = READERS[ jObj[DT_JSON_OCC_READER] ]()

                        # Create a testing dataset.
                        datasetTest = make_object(
                            DATASET_LOADERS, self.loaderConfMain['loaderTest'])
                        datasetTest.set_lists(
                            left=imgL, right=imgR, dispL=disp)
                        datasetTest.set_readers(
                            imgReader=imgLoader, dispReader=dispLoader,
                            occReader=occLoader )

                        if ( DT_JSON_OCC_TEST in jObj and jObj[DT_JSON_OCC_TEST] != '' ):
                            [occL] = read_string_list_2D( \
                                jObj[DT_JSON_FL_DIR] + "/" + jObj[DT_JSON_OCC_TEST], \
                                1, delimiter=jObj[DT_JSON_DLMT], prefix=jObj[DT_JSON_DT_ROOT] )

                            assert( len(occL) == len(imgL) )
                            datasetTest.set_occ_L( occL )

                        if ( DT_JSON_VALID_TEST in jObj and jObj[DT_JSON_VALID_TEST] != ''):
                            [validMask] = read_string_list_2D(
                                jObj[DT_JSON_FL_DIR] + "/" + jObj[DT_JSON_VALID_TEST],
                                1, delimiter=jObj[DT_JSON_DLMT], prefix=jObj[DT_JSON_DT_ROOT] )

                            assert( len(validMask) == len(imgL) )
                            datasetTest.set_valid_mask( validMask )
                        
                        # # Flag for occ.
                        # if ( jObj[DT_JSON_USE_OCC] ):
                        #     datasetTest.flagUseOcc = True
                        # else:
                        #     datasetTest.flagUseOcc = False

                        testDatasets.append( datasetTest )
            except Exception as exp:
                print('Exception catched while processing %s. ' % (fn) )
                print('Re-raise the exception')
                raise exp

        return trainDatasets, testDatasets

    def find_and_create_data_representation(self):
        print('Try to find and in-training test datasets. ')

        drs = []   
        for datasetDict in self.datasetJSONList:
            fn = datasetDict['description']
            try:
                if ( not os.path.isfile(fn) ):
                    raise Exception("{} not exist. ".format(fn))

                # Read the json file.
                jObj = read_json(fn)
                if ( not 'flagTestInTraining' in jObj.keys() or not jObj['flagTestInTraining'] ):
                    continue

                # Found a in-training test dataset.
                drs.append( create_data_representation_from_json_obj(jObj) )
            except Exception as exp:
                print(f'Error occurs when parsing {fn}. ')
                raise exp
        
        return drs

    def parse(self):
        trainDatasets, testDatasets = \
            self.create_datasets_from_json_list()

        # Merge all the available datasets.
        trainMergedDataset = torch.utils.data.ConcatDataset( trainDatasets )
        self.loaderTrain = torch.utils.data.DataLoader(
            trainMergedDataset,
            batch_size=self.batchSize, shuffle=self.shuffle, 
            num_workers=self.numWorkers, drop_last=self.dropLast )

        print("Total training samples: %d. " % ( len(trainMergedDataset) ))

        if ( self.flagTest ):
            testBatchSize = self.batchSize
        else:
            testBatchSize = int( max(self.batchSize*self.testBatchFactor, 1) )

        testMergedDataset = torch.utils.data.ConcatDataset( testDatasets )
        self.loaderTest = torch.utils.data.DataLoader(
            testMergedDataset,
            batch_size=testBatchSize, shuffle=False, 
            num_workers=self.numWorkers, drop_last=self.dropLast )

        print("Total testing samples:  %d. " % ( len(testMergedDataset) ))

        # In-training test dataset.
        drs = self.find_and_create_data_representation()
        if ( len(drs) > 0 ):
            self.dataITT = make_object(DATASET_LOADERS, self.loaderConfITT)
            print('In-training test datasets found. ')
            
            self.dataITT.preload(drs)

            self.loaderITT = torch.utils.data.DataLoader(
                self.dataITT, batch_size=1, shuffle=False, 
                num_workers=self.numWorkers, drop_last=False )
        else:
            print('In-training test datasets not found. ')
            self.loaderITT = None
