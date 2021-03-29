# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-23

import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
# if ( not ( "DISPLAY" in os.environ ) ):
#     plt.switch_backend('agg')
#     print("TTNG: Environment variable DISPLAY is not present in the system.")
#     print("TTNG: Switch the backend of matplotlib to agg.")
plt.switch_backend('agg')
print("Switch the backend of matplotlib to agg.")

import torch
import torch.nn.functional as F
import torch.optim as optim

# Local installed modules.
from workflow import WorkFlow, TorchFlow
from CommonPython.PointCloud.PLYHelper import write_PLY

from stereo.dynamic_config.config import config_2_str

from stereo.models.register import( MODELS, LOSS_CMP, TRUE_GEN, make_object )
from stereo.learning_rate_scheduler.register import ( LR_SCHEDULERS, make_scheduler )
from stereo.models.supervision.true_value import TT
from stereo.models.supervision.basic_losses import LT
from stereo.metric.register import( METRICS, make_metric )
from stereo.visualization import tensor_handles
from stereo.visualization.test_result_drawing_tools import compose_results

from .tt_base import TrainTestBase

Q_FLIP = np.array( [ \
    [ 1.0,  0.0,  0.0, 0.0 ], \
    [ 0.0, -1.0,  0.0, 0.0 ], \
    [ 0.0,  0.0, -1.0, 0.0 ], \
    [ 0.0,  0.0,  0.0, 1.0 ] ], dtype=np.float32 )

def single_channel_mask_from_four_channel(a, m):
    '''It is assumed that m has dimension [B, H, W]'''
    b = a.permute((1, 0, 2, 3))
    return b[:, m].unsqueeze(0)

class TTS(TrainTestBase):
    def __init__(self, workingDir, config, frame=None):
        super(TTS, self).__init__( workingDir, config, frame )

        self.metric = None

    # def initialize(self):
    #     self.check_frame()
    #     raise Exception("Not implemented.")

    # Overload parent's function.
    def init_workflow(self):
        # === Create the AccumulatedObjects. ===
        self.frame.add_accumulated_value("L0", 50)
        self.frame.add_accumulated_value("LT", 10)
        self.frame.add_accumulated_value("LT0", 10)
        self.frame.add_accumulated_value("TDA", 1)
        self.frame.add_accumulated_value('lr1000')

        self.frame.AV["loss"].avgWidth = 10
        
        # ======= AVP. ======
        # === Create a AccumulatedValuePlotter object for ploting. ===
        if ( True == self.flagUseIntPlotter ):
            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.frame.workingDir + "/IntPlot", 
                    "loss", self.frame.AV, ["loss", "L0"], [True, True], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.frame.workingDir + "/IntPlot", 
                    "ITT", self.frame.AV, ["L0", "LT0"], [True, True], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.frame.workingDir + "/IntPlot", 
                    "LT", self.frame.AV, ["LT", "LT0"], [False, False], semiLog=True) )
            
            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.frame.workingDir + "/IntPlot", 
                    "TDA", self.frame.AV, ["TDA"], [False], semiLog=False) )

            self.frame.AVP.append(\
                WorkFlow.PLTIntermittentPlotter(\
                    self.frame.workingDir + "/IntPlot", 
                    'lr1000', self.frame.AV, ['lr1000'], [False], semiLog=True) )
        else:
            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "loss", self.frame.AV, ["loss", "L0"], [True, True], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "ITT", self.frame.AV, ["L0", "LT0"], [True, True], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "LT", self.frame.AV, ["LT", "LT0"], [False, False], semiLog=True) )

            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    "TDA", self.frame.AV, ["TDA"], [False], semiLog=False) )

            self.frame.AVP.append(\
                WorkFlow.VisdomLinePlotter(\
                    'lr1000', self.frame.AV, ['lr1000'], [False], semiLog=True) )

    # def init_workflow(self):
    #     raise Exception("Not implemented.")

    # def init_torch(self):
    #     raise Exception("Not implemented.")

    # def init_data(self):
    #     raise Exception("Not implemented.")

    def show_configuration_details(self, config):
        self.check_frame()

        self.frame.logger.info('===== Settings in the config file. =====')
        self.frame.logger.info( config_2_str(self.conf) )
        
    # Overload parent's function.
    def init_model(self):
        conf = self.conf
        
        # Show configuration details.
        self.show_configuration_details( conf )

        # Configure align corners globaly.
        if ( conf['globals']['alignCorners'] ):
            self.enable_align_corners()
        else:
            self.disable_align_corners()

        if ( conf['globals']['trackRunningStats'] ):
            self.enable_batch_norm_track_running_stat()
        else:
            self.disable_batch_norm_track_running_stat()

        # Configure the model.
        self.model = make_object( MODELS, conf['tt']['model'] )
        self.model.initialize()

        # Check if we have to read the model from filesystem.
        if ( "" != self.readModelString ):
            modelFn = self.frame.workingDir + "/models/" + self.readModelString

            if ( False == os.path.isfile( modelFn ) ):
                raise Exception("Model file (%s) does not exist." % ( modelFn ))

            # self.model = self.frame.load_model( self.model, modelFn )
            self.frame.load_model( self.model, modelFn )

        # if ( self.flagCPU ):
        #     self.model.set_cpu_mode()

        self.frame.logger.info("Model %s has %d parameters." % \
            ( self.modelName, sum( [ p.data.nelement() for p in self.model.parameters() ] ) ) )
    
    # def post_init_model(self):
    #     raise Exception("Not implemented.")

    def update_learning_rate(self):
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.learningRate

    # Overload parent's function.
    def init_optimizer(self):
        # self.optimizer = optim.Adam( self.model.parameters(), lr=0.001, betas=(0.9, 0.999) )
        if ( "adam" == self.optType ):
            self.optimizer = optim.Adam( self.model.parameters(), lr=self.learningRate )
        elif ( "sgd" == self.optType ):
            self.optimizer = optim.SGD( self.model.parameters(), lr=self.learningRate )
        else:
            raise Exception("Unexpected optimizer type: {}. ".format(self.optType))

        # Check if we have to read the optimizer state from the filesystem.
        if ( "" != self.readOptimizerString ):
            optFn = "%s/models/%s" % ( self.frame.workingDir, self.readOptimizerString )

            if ( not os.path.isfile( optFn ) ):
                raise Exception("Optimizer file (%s) does not exist. " % ( optFn ))

            # self.optimizer = self.frame.load_optimizer(self.optimizer, optFn)
            self.frame.load_optimizer(self.optimizer, optFn)
            # Update the learning rate.
            self.update_learning_rate()
            self.frame.logger.info("Optimizer state loaded for file %s. " % (optFn))

        # Learning rate scheduler.
        if ( self.flagUseLRS ):
            # Choose a scheduler.
            self.lrs = make_scheduler( 
                LR_SCHEDULERS, self.optimizer, self.conf['tt']['lrs'] )
            self.frame.logger.info("Configured learning rate scheduler %s. " % ( self.conf['tt']['lrs']['type'] ))
            self.frame.logger.info(str(self.lrs))

        # Loss computer.
        self.lossComputer = \
            make_object(LOSS_CMP, self.conf['tt']['lossComputer'])

    def init_data(self):
        super(TTS, self).init_data()
        self.trueValueGenerator = \
            make_object(TRUE_GEN, self.conf['tt']['trueValueGenerator'])
        
        self.metric = make_metric( METRICS, self.conf['tt']['metric'] )

    def interpolate(self, t, shape):
        if ( self.flagIntNearest ):
            return F.interpolate( t, shape, 
                mode="nearest" )
        else:
            return F.interpolate( t, shape, 
                mode="bilinear", align_corners=self.flagAlignCorners )

    # Overload parent's function.
    def train(self, inputs, epochCount):
        self.check_frame()
        self.model.train()

        imgL   = inputs['img0']
        imgR   = inputs['img1']
        dispL  = inputs['disp0']
        validL = inputs['valid0'] if 'valid0' in inputs else None

        if ( not self.flagCPU ):
            imgL  = imgL.cuda()
            imgR  = imgR.cuda()
            dispL = dispL.cuda()

            if ( validL is not None):
                validL = validL.cuda()

        self.optimizer.zero_grad()

        # Forward.
        outputs = self.model( { 'img0':imgL, 'img1':imgR } )
        
        # Create a set of true values.
        trueValueDict = self.trueValueGenerator.make_true_values(
            dispL, None, validL)

        # Compute loss.
        lossDict = self.lossComputer.compute_loss( trueValueDict, outputs )
        loss = lossDict[LT.LOSS]

        # Backward.
        loss.backward()

        self.optimizer.step()

        # LRS.
        if ( self.flagUseLRS ):
            self.lrs.step(loss)

        self.frame.AV['loss'].push_back( loss.item() )
        self.frame.AV['L0'].push_back( lossDict[LT.LOSS_LIST][0].item() )
        if ( validL is not None ):
            self.frame.AV["TDA"].push_back( dispL[validL].mean().item() )
        else:
            self.frame.AV["TDA"].push_back( dispL.mean().item() )
        self.frame.AV['lr1000'].push_back( self.optimizer.param_groups[0]['lr'] * 1000 )

        self.countTrain += 1

        if ( self.countTrain % self.trainIntervalAccWrite == 0 ):
            self.frame.write_accumulated_values()

        # Plot accumulated values.
        if ( self.countTrain % self.trainIntervalAccPlot == 0 ):
            self.frame.plot_accumulated_values()

        # Auto-save.
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

        self.frame.logger.info("E%d, L%d: %s" % (epochCount, self.countTrain, self.frame.get_log_str()))

    def draw_test_results(self, identifier,
            imgL, imgR,
            trueDList, predDList ):
        batchSize = predDList[0].shape[0]
        
        img0List = tensor_handles.batch_img_2_array_list(imgL)
        img1List = tensor_handles.batch_img_2_array_list(imgR)

        for i in range(batchSize):
            img0 = tensor_handles.simple_array_2_rgb(img0List[i])
            img1 = tensor_handles.simple_array_2_rgb(img1List[i])

            trueDispList = []
            predDispList = []
            
            for j in range( len(trueDList) ):
                trueDispList.append( trueDList[j][i, 0, :, :].detach().cpu().numpy() )
                predDispList.append( predDList[j][i, 0, :, :].detach().cpu().numpy() )

            canvas = compose_results( 
                img0, img1, trueDispList, predDispList )

            figName = "%s_%02d" % (identifier, i)
            figName = self.frame.compose_file_name(figName, "png", subFolder=self.testResultSubfolder)

            cv2.imwrite(figName, canvas)

    def save_test_disp(self, identifier, pred):
        batchSize = pred.size()[0]
        
        for i in range(batchSize):
            # Get the prediction.
            disp = pred[i, 0, :, :].cpu().numpy()

            fn = "%s_%02d" % (identifier, i)
            fn = self.frame.compose_file_name(fn, "npy", subFolder=self.testResultSubfolder)

            # np.savetxt( fn, disp, fmt="%+.6f" )
            np.save( fn, disp )

    # Overload parent's function.
    def test(self, inputs, epochCount, flagSave=True, flagInTrainingTest=False):
        self.check_frame()
        self.model.eval()

        imgL   = inputs['img0']
        imgR   = inputs['img1']
        dispL  = inputs['disp0']
        validL = inputs['valid0'] if 'valid0' in inputs else None

        if ( not self.flagCPU ):
            imgL  = imgL.cuda()
            imgR  = imgR.cuda()
            dispL = dispL.cuda()

            if ( validL is not None ):
                validL = validL.cuda()

        # Create a set of true data with various scales.
        trueValueDict = self.trueValueGenerator.make_true_values(\
            dispL, None, validL)

        with torch.no_grad():
            outputs = self.model( { 'img0':imgL, 'img1':imgR } )
        
        predDispList = outputs[TT.DISP_LIST]
        disp0 = predDispList[0]
            
        if ( not flagInTrainingTest ):
            with torch.no_grad():
                lossDict = self.lossComputer.compute_loss( trueValueDict, outputs )
                loss = lossDict[LT.LOSS]

                # Apply metrics.
                dispLNP = dispL.squeeze(1).cpu().numpy()
                mask    = dispLNP <= self.trueValueGenerator.trueDispMax
                # mask    = mask.astype(np.int)
                metrics = self.metric.apply( dispLNP, disp0.squeeze(1).cpu().numpy(), mask )
                
            self.countTest += 1

            if ( True == self.flagTest ):
                count = self.countTest
            else:
                count = self.countTrain

            if ( flagSave ):
                # Draw and save results.
                identifier = "test_%d" % (count - 1)
                self.save_test_disp( identifier, disp0 )

                self.draw_test_results( identifier, 
                    imgL, imgR, \
                    [ dispL ] * len(predDispList), \
                    predDispList )

            # AccumulatedValue objects.
            self.frame.AV["LT"].push_back(loss.item(), self.countTest)
            self.frame.AV["LT0"].push_back(lossDict[LT.LOSS_LIST][0].item(), self.countTest)

            if ( flagSave ):
                self.frame.plot_accumulated_values()

            return loss.item(), metrics
        else:
            mask0 = trueValueDict[TT.MASK]
            with torch.no_grad():
                loss = F.l1_loss( disp0[mask0], dispL[mask0] , reduction="mean" )
            return loss.item()

    def infer(self, imgL, imgR, gradL, gradR, Q):
        raise NotImplementedError()

    # Overload parent's function.
    def finalize(self):
        self.check_frame()

        # Save the model and optimizer.
        if ( False == self.flagTest ):
            self.frame.save_model( self.model, self.modelName )
            self.frame.save_optimizer( self.optimizer, "%s_Opt" % (self.modelName) )
        # self.frame.logger.warning("Model not saved for dummy test.")
