# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-23

import json
import numpy as np
import os
import sys
import time

from workflow import WorkFlow, TorchFlow

from stereo.application import arg_utils
from stereo.application import arg_parser
from stereo.application.tts import TTS

import faulthandler
faulthandler.enable()

def print_delimeter(c = "=", n = 20, title = "", leading = "\n", ending = "\n"):
    d = [c for i in range( int(n/2) )]

    if ( 0 == len(title) ):
        s = "".join(d) + "".join(d)
    else:
        s = "".join(d) + " " + title + " " + "".join(d)

    print("%s%s%s" % (leading, s, ending))

# Template for custom WorkFlow object.
class MyWF(TorchFlow.TorchFlow):
    def __init__(self, workingDir, prefix = "", suffix = "", disableStreamLogger=False):
        super(MyWF, self).__init__(workingDir, prefix, suffix, disableStreamLogger)
        # === Custom member variables. ===
        self.tt = None # The TrainTestBase object.
        self.timeStart = None

    def set_tt(self, tt):
        self.tt = tt

    def check_tt(self):
        if ( self.tt is None ):
            Exception("self.tt must not be None.")

    # Overload the function initialize().
    def initialize(self):
        super(MyWF, self).initialize()
        self.check_tt()

        # === Custom code. ===
        self.tt.initialize()
        self.timeStart = time.time()
        self.logger.info("Initialized.")
        self.post_initialize()

    # Overload the function train().
    def train(self, inputs, epochCount):
        super(MyWF, self).train()
        self.check_tt()
        return self.tt.train(inputs, epochCount)
        
    # Overload the function test().
    def test(self, inputs, epochCount, flagSave=True, flagInTrainingTest=False):
        super(MyWF, self).test()
        self.check_tt()
        return self.tt.test(inputs, epochCount, flagSave, flagInTrainingTest)

    def infer(self, imgL, imgR, gradL, Q):
        raise NotImplementedError()

    # Overload the function finalize().
    def finalize(self):
        super(MyWF, self).finalize()
        self.check_tt()

        # Calculate the total time.
        timeEnd = time.time()
        self.logger.info("Workflow execute in %fh. " %\
            ( (timeEnd - self.timeStart) / 3600.0 ))
        self.tt.finalize()
        self.logger.info("Finalized.")

def get_average_epe_by_pixel_number(testMetricList):
    allEPE = testMetricList[:, 5] * testMetricList[:, 6]
    return allEPE.sum() / testMetricList[:, 6].sum()

def main(args, sc):
    dataJSONList = arg_utils.read_string_list(args.data_json_list)

    # Update sc.
    sc['tt']['dataloader']['datasetJSONList'] = dataJSONList

    print_delimeter(title = "Before WorkFlow initialization." )

    try:
        # Instantiate an object for MyWF.
        wf = MyWF(args.working_dir, prefix=args.prefix, suffix=args.suffix, disableStreamLogger=False)
        wf.verbose = False

        # Cross reference.
        tt = TTS(wf.workingDir, sc, wf)
        wf.set_tt(tt)
        
        tt.set_model_name(args.model_name)

        if ( True == args.multi_gpus ):
            tt.enable_multi_GPUs()

        tt.set_read_optimizer( args.read_optimizer )

        # Model
        tt.set_read_model( args.read_model )
        tt.enable_auto_save( args.auto_save_model )
        tt.set_auto_snap_loops( args.auto_snap_loops )
        tt.set_training_acc_params( args.train_interval_acc_write, args.train_interval_acc_plot, args.use_intermittent_plotter )

        if ( True == args.test ):
            tt.switch_on_test()
        else:
            tt.switch_off_test()

        # Initialization.
        print_delimeter(title = "Initialize.")
        wf.initialize()

        # Get the number of test data.
        nTests = len( tt.dataloader.loaderTest )
        wf.logger.info("The size of the test dataset is %s." % ( nTests ))

        if ( not args.test ):
            if ( tt.dataloader.loaderITT is None ):
                # Create the test data iterator.
                iterTestData = iter( tt.dataloader.loaderTest )

                # Training loop.
                wf.logger.info("Begin training.")
                print_delimeter(title="Training loops.")

                dataDictTest = next( iterTestData )
                wf.test( dataDictTest, 0 )

                for i in range(args.train_epochs):
                    for batchIdx, dataDictTrain in enumerate( tt.dataloader.loaderTrain ):
                        
                        wf.train( dataDictTrain, i )

                        # Check if we need a test.
                        if ( 0 != args.test_loops ):
                            if ( tt.countTrain % args.test_loops == 0 ):
                                # Get test data.
                                try:
                                    dataDictTest = next( iterTestData )
                                except StopIteration:
                                    iterTestData = iter(tt.dataloader.loaderTest)
                                    dataDictTest = next( iterTestData )

                                # Perform test.
                                wf.test( dataDictTest, i )
            else:
                # ITT.
                wf.logger.info('Initial ITT...')

                testLosses = []
                for dataDictTest in tt.dataloader.loaderITT:
                    loss = wf.test( dataDictTest, 0, flagSave=False, flagInTrainingTest=True )
                    testLosses.append(loss)
                
                # compute the average loss.
                meanLoss = np.mean( testLosses )
                tt.countTest += 1
                tt.frame.AV["LT0"].push_back(meanLoss, tt.countTest)

                print_delimeter(title="Training loops.")

                for i in range(args.train_epochs):
                    for batchIdx, dataDictTrain in enumerate( tt.dataloader.loaderTrain ):
                        
                        wf.train( dataDictTrain, i )

                        # Check if we need a test.
                        if ( 0 != args.test_loops ):
                            if ( tt.countTrain % args.test_loops == 0 ):
                                wf.logger.info('ITT...')

                                testLosses = []
                                for dataDictTest in tt.dataloader.loaderITT:
                                    loss = wf.test( dataDictTest, 0, flagSave=False, flagInTrainingTest=True )
                                    testLosses.append(loss)
                                
                                # compute the average loss.
                                meanLoss = np.mean( testLosses )
                                tt.countTest += 1
                                tt.frame.AV["LT0"].push_back(meanLoss, tt.countTrain)

        else:
            wf.logger.info("Begin testing.")
            print_delimeter(title="Testing loops.")

            testLossList = []
            testMetricList = np.array([], dtype=np.float32).reshape((0, 7)) 

            for batchIdx, dataDictTest in enumerate( tt.dataloader.loaderTest ):
                loss, metrics = wf.test( dataDictTest, batchIdx, args.test_flag_save )

                if ( True == tt.flagInspect ):
                    wf.logger.warning("Inspection enabled.")

                wf.logger.info("Test %d, lossTest = %f." % ( batchIdx, loss ))

                testLossList.append( [ dataDictTest["img0"].size()[0], loss, *(np.mean(metrics, axis=0).tolist()) ] )

                testMetricList = np.vstack((testMetricList, metrics))

                if ( args.test_loops > 0 and batchIdx >= args.test_loops - 1 ):
                    break

            testLossAndMetrics = np.array(testLossList, dtype=np.float32)
            scaledLossAndMetrics = testLossAndMetrics[:, 1:] * testLossAndMetrics[:, 0].reshape((-1,1))
            finiteMask = np.isfinite( scaledLossAndMetrics )
            finiteMask = np.all( finiteMask, axis=1 )
            averagedLossAndMetrics = np.sum(scaledLossAndMetrics[finiteMask, :], axis=0) / testLossAndMetrics[:, 0].sum()

            wf.logger.info("Average loss = %f." % ( averagedLossAndMetrics[0] ))
            wf.logger.info("Average 1-pixel error rate = %f." % ( averagedLossAndMetrics[1] ))
            wf.logger.info("Average 2-pixel error rate = %f." % ( averagedLossAndMetrics[2] ))
            wf.logger.info("Average 3-pixel error rate = %f." % ( averagedLossAndMetrics[3] ))
            wf.logger.info("Average 4-pixel error rate = %f." % ( averagedLossAndMetrics[4] ))
            wf.logger.info("Average 5-pixel error rate = %f." % ( averagedLossAndMetrics[5] ))
            wf.logger.info("Average end point error = %f." % ( averagedLossAndMetrics[6] ))

            # The EPE calculated by revealing the pixel numbers.
            avgEPEByPixelNumber = get_average_epe_by_pixel_number(testMetricList)
            wf.logger.info("Average end point error by pixel number = %f. " % ( avgEPEByPixelNumber ))

            # Save the loss values to file the working directory.
            testResultSummaryFn = wf.compose_file_name("BatchTest", "dat", subFolder=tt.testResultSubfolder)
            np.savetxt( testResultSummaryFn, testLossList)
            testMetricListFn = wf.compose_file_name("BatchTestMetrics", "dat", subFolder=tt.testResultSubfolder)
            np.savetxt( testMetricListFn, testMetricList )

        wf.finalize()
    except WorkFlow.SigIntException as sie:
        print("SigInt revieved, perform finalize...")
        wf.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")

    return 0

if __name__ == "__main__":
    print("Hello, %s! " % ( os.path.basename( __file__ ) ))

    # Handle the arguments.
    args = arg_parser.args

    # Modify the python search path on the fly.
    sys.path.insert(0, args.working_dir)

    # Read the session configuration file.
    # input_conf.py must be present in the working directory. 
    from input_conf import conf as sessionConfig

    sys.exit( main( args, sessionConfig ) )
