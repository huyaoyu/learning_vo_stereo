# -*- coding: future_fstrings -*-

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-22

# System packages.
import copy
import json
import numpy as np
import os

# PyTorch packages.
import torch
import torch.nn as nn

# Local installed.
from workflow import WorkFlow, TorchFlow

# Models.
from stereo.models.globals import GLOBAL as modelGLOBAL

from stereo.dataset.compound_loader import CompoundLoader
from stereo.visualization.figure_grid import FigureGrid

RECOMMENDED_MIN_INTERMITTENT_PLOT_INTERVAL = 100

class InfoUpdater(object):
    def __init__(self, ctxHandel, name, func):
        super(InfoUpdater, self).__init__()

        self.ctx  = ctxHandel
        self.name = name
        self.func = func

    def __call__(self):
        self.ctx.frame.AV[self.name].push_back( 
            self.func( self.ctx ) )

class TrainTestBase(object):
    def __init__(self, workingDir, conf, frame=None, modelName='Stereo'):
        self.conf = conf # The configuration dictionary.
        
        self.wd        = workingDir
        self.frame     = frame
        self.modelName = modelName

        # NN.
        self.countTrain = 0
        self.countTest  = 0

        self.flagAlignCorners = modelGLOBAL.torch_align_corners()
        self.flagIntNearest   = False

        self.trainIntervalAccWrite = 10    # The interval to write the accumulated values.
        self.trainIntervalAccPlot  = 1     # The interval to plot the accumulate values.
        self.flagUseIntPlotter     = False # The flag of intermittent plotter.

        self.flagCPU   = False
        self.multiGPUs = False

        self.readModelString     = ""
        self.readOptimizerString = ""
        self.autoSaveModelLoops  = 0 # The number of loops to perform an auto-saving of the model. 0 for disable.
        self.autoSnapLoops       = 100 # The number of loops to perform an auto-snap.

        self.optimizer = None

        self.flagTest  = False # Should be set to True when testing.

        self.flagRandomSeedSet = False

        # Specified by conf.
        self.trueDispMask        = conf['tt']['trueDispMask']
        self.model               = None # make_object later.
        self.dataloader          = None # make_object later.
        self.optType             = conf['tt']['optType'] # The optimizer type. adam, sgd.
        # Learning rate scheduler.
        self.flagUseLRS          = conf['tt']['flagUseLRS']
        self.learningRate        = conf['tt']['lr']
        self.lrs                 = None # make_object later.
        # True value and loss.
        self.trueValueGenerator  = None # make_object later.
        self.lossComputer        = None # make_object later.
        self.testResultSubfolder = conf['tt']['testResultSubfolder']
        # Test figure generator.
        self.testFigGenerator = None

        # Temporary values during traing and testing.
        self.ctxInputs     = None
        self.ctxOutputs    = None
        self.ctxTrueValues = None
        self.ctxLossValues = None

        # InfoUpdaters.
        self.trainingInfoUpdaters = []
        self.testingInfoUpdaters  = []

    def initialize(self):
        self.check_frame()

        # Over load these functions if nessesary.
        self.init_base()
        self.init_workflow()
        self.init_torch()
        self.init_data()
        self.init_model()
        self.post_init_model()
        self.init_optimizer()
        self.init_test_figure_generator()
    
    def train(self):
        self.check_frame()
    
    def test(self):
        self.check_frame()
    
    def in_training_test(self):
        self.check_frame()

    def finialize(self):
        self.check_frame()

    def infer(self):
        self.check_frame()

    def set_frame(self, frame):
        self.frame = frame
    
    def check_frame(self):
        if ( self.frame is None ):
            raise Exception("self.frame must not be None.")
    
    def set_model_name(self, name):
        self.modelName = name

    def enable_align_corners(self):
        self.check_frame()
        self.frame.logger.info("Use align_corners=True.")
        modelGLOBAL.torch_align_corners(True)
        self.flagAlignCorners = True
    
    def disable_align_corners(self):
        self.check_frame()
        self.frame.logger.info("Use align_corners=False.")
        modelGLOBAL.torch_align_corners(False)
        self.flagAlignCorners = False

    def enable_batch_norm_track_running_stat(self):
        self.check_frame()
        self.frame.logger.info('Use track_running_stats=True.')
        modelGLOBAL.torch_batch_normal_track_stat(True)

    def disable_batch_norm_track_running_stat(self):
        self.check_frame()
        self.frame.logger.info('Use track_running_stats=False.')
        modelGLOBAL.torch_batch_normal_track_stat(False)

    def set_padding_mode(self, mode):
        self.check_frame()
        self.frame.logger.info(f'Set padding mode to {mode}')
        modelGLOBAL.padding_mode(mode)

    def enable_nearest_interpolation(self):
        self.check_frame()
        self.frame.logger.info("Use nearest interpolation for true values. ")
        self.flagIntNearest = True

    def enable_last_regression_kernel_size_one(self):
        self.check_frame()
        self.frame.logger.info("Use last regression kernel size one. ")
        modelGLOBAL.last_regression_kernel_size_one(True)

    def enable_multi_GPUs(self):
        self.check_frame()

        self.flagCPU   = False
        self.multiGPUs = True

        if ( self.flagRandomSeedSet ):
            self.set_random_seed()

        self.frame.logger.info("Enable multi-GPUs.")

    def set_cpu_mode(self):
        self.check_frame()

        self.flagCPU   = True
        self.multiGPUs = False

        self.frame.logger.warning("CPU mode is selected.")

    def unset_cpu_mode(self):
        self.check_frame()

        self.flagCPU   = False
        self.multiGPUs = False

        self.frame.logger.warning("Back to GPU mode.")

    def set_read_model(self, readModelString):
        self.check_frame()
        
        self.readModelString = readModelString

        if ( "" != self.readModelString ):
            self.frame.logger.info("Read model from %s." % ( self.readModelString ))

    def set_read_optimizer(self, readOptimizerString):
        self.check_frame()

        self.readOptimizerString = readOptimizerString

        if ( "" != self.readOptimizerString ):
            self.frame.logger.info("Read optimizer from %s. " % ( self.readOptimizerString ))

    def enable_auto_save(self, loops):
        self.check_frame()
        
        self.autoSaveModelLoops = loops

        if ( 0 != self.autoSaveModelLoops ):
            self.frame.logger.info("Auto save enabled with loops = %d." % (self.autoSaveModelLoops))

    def set_auto_snap_loops(self, loops):
        self.check_frame()

        assert(loops > 0)

        self.autoSnapLoops = loops

    def set_training_acc_params(self, intervalWrite, intervalPlot, flagInt=False):
        self.check_frame()
        
        self.trainIntervalAccWrite = intervalWrite
        self.trainIntervalAccPlot  = intervalPlot
        self.flagUseIntPlotter     = flagInt

        if ( True == self.flagUseIntPlotter ):
            if ( self.trainIntervalAccPlot <= RECOMMENDED_MIN_INTERMITTENT_PLOT_INTERVAL ):
                self.frame.logger.warning("When using the intermittent plotter. It is recommended that the plotting interval (%s) is higher than %d." % \
                    ( self.trainIntervalAccPlot, RECOMMENDED_MIN_INTERMITTENT_PLOT_INTERVAL ) )

    def switch_on_test(self):
        self.flagTest = True

    def switch_off_test(self):
        self.flagTest = False

    def set_random_seed(self):
        torch.random.seed()

        if ( self.multiGPUs ):
            torch.cuda.random.seed_all()
        else:
            torch.cuda.random.seed()

    def init_base(self):
        # Make the subfolder for the test results.
        self.frame.make_subfolder(self.testResultSubfolder)

    def init_workflow(self):
        raise Exception("init_workflow() virtual interface.")

    def init_torch(self):
        self.check_frame()

        self.frame.logger.info("Configure Torch.")

        # torch.manual_seed(1)
        # torch.cuda.manual_seed(1)

    def init_data(self):
        conf = self.conf['tt']['dataloader']
        # Dataloader.
        self.dataloader = CompoundLoader(
            conf['datasetJSONList'], 
            batchSize=conf['batchSize'], 
            shuffle=conf['shuffle'],
            numWorkers=conf['numWorkers'],
            dropLast=conf['dropLast'],
            testBatchFactor=conf['testBatchFactor'],
            loaderConfMain=conf['loaderConfMain'],
            loaderConfITT=conf['loaderConfITT'] )

        if ( self.flagTest ):
            self.dataloader.set_test()

        self.dataloader.parse()

    def init_model(self):
        raise Exception("init_model() virtual interface.")

    def post_init_model(self):
        if ( not self.flagCPU ):
            if ( True == self.multiGPUs ):
                self.model = nn.DataParallel(self.model)

            self.model.cuda()
    
    def set_optimizer_type(self, t):
        self.optType = t

    def enable_learning_rate_scheduler(self):
        self.flagUseLRS = True

    def disable_learning_rate_scheduler(self):
        self.flagUseLRS = False

    def init_optimizer(self):
        raise Exception("init_optimizer() virtual interface.")

    def init_test_figure_generator(self):
        self.testFigGenerator = FigureGrid( **self.conf['tt']['testFigGenerator'] )

    def register_running_info(self, phase, name, avgSteps, func):
        assert( avgSteps >= 1 )
        self.frame.add_accumulated_value(name, avgSteps, overwrite=True)
        if ( phase == 'training' ):
            self.trainingInfoUpdaters.append( InfoUpdater( self, name, func ) )
        elif ( phase == 'testing' ):
            self.testingInfoUpdaters.append( InfoUpdater( self, name, func ) )

    def register_info_plotter(self, name, infoNames, avgFlags, subDir='IntPlot', flagSemiLog=False):
        if ( self.flagUseIntPlotter ):
            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    os.path.join(self.frame.workingDir, subDir), 
                    name, self.frame.AV, infoNames, avgFlags, semiLog=flagSemiLog) )
        else:
            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    name, self.frame.AV, infoNames, avgFlags, semiLog=flagSemiLog) )

    def update_running_info(self, training=True):
        updaters = self.trainingInfoUpdaters if training else self.testingInfoUpdaters
        for updater in updaters:
            updater()

    def save_running_info(self, training=True):
        if ( training ):
            # Save the values.
            if ( self.countTrain % self.trainIntervalAccWrite == 0 ):
                self.frame.write_accumulated_values()

            # Plot accumulated values.
            if ( self.countTrain % self.trainIntervalAccPlot == 0 ):
                self.frame.plot_accumulated_values()
        else:
            self.frame.plot_accumulated_values()

    def auto_save(self):
        if ( 0 != self.autoSaveModelLoops ):
            if ( self.countTrain % self.autoSaveModelLoops == 0 ):
                modelName = "AutoSave_%08d" % ( self.countTrain )
                optName   = "AutoSave_Opt_%08d" % ( self.countTrain )
                self.frame.logger.info("Auto-save the model and optimizer.")
                self.frame.save_model( self.model, modelName )
                self.frame.save_optimizer( self.optimizer, optName )

        if ( self.countTrain % self.autoSnapLoops == 0 ):
            modelName = "AutoSnap"
            optName   = "AutoSnap_Opt"
            self.frame.save_model( self.model, modelName )
            self.frame.save_optimizer( self.optimizer, optName )