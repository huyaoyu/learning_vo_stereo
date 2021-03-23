DISP_MAX=192

conf=dict(
    globals=dict(
        alignCorners=True,
        trackRunningStats=True,
        reluInPlace=False ),
    tt=dict(
        trueDispMask=DISP_MAX,
        model=dict(
            type='HSMNet',
            maxDisp=DISP_MAX,
            level=1,
            featExtConfig=dict(
                type='UNet',
                initialChannels=32, freeze=False),
            costVolConfig=dict(
                type='CVDiff',
                refIsRight=False, freeze=False),
            dispRegConfig=dict(
                type='ClsLinearCombination', 
                maxDisp=DISP_MAX, divisor=1, freeze=False),
            freeze=False),
        dataloader=dict(
            datasetJSONList=[], # Get updated when used.
            batchSize=2, 
            shuffle=True, 
            numWorkers=2, 
            dropLast=False,
            testBatchFactor=1.0,
            loaderConfMain=dict(
                loaderTrain=dict(
                    type='FileListLoader',
                    flagUseOcc=False,
                    ppConfs=[
                        dict(
                            type='MeanDispBasedRandomScale_OCV_Dict',
                            maxDisp=DISP_MAX, flagRandom=True, randomLimit=0.5),
                        dict(
                            type='RandomCropSized_OCV_Dict',
                            h=448, w=640),
                        dict(
                            type='VerticalFlip_OCV_Dict',
                            flagRandom=True, randomLimit=0.5),
                        dict(
                            type='RandomColor_OCV_Dict',
                            hSpan=25, sSpan=50, vSpan=50, scale=0.2,
                            flagRandom=True, randomLimit=0.5),
                        dict(
                            type='NormalizeRGB_OCV_Dict',
                            s=255),
                        dict(
                            type='ToTensor_OCV_Dict')],
                    training=True),
                loaderTest=dict(
                    type='FileListLoader',
                    flagUseOcc=False,
                    ppConfs=[
                        dict(
                            type='CenterCropSized_OCV_Dict',
                            h=448, w=640),
                        dict(
                            type='NormalizeRGB_OCV_Dict',
                            s=255),
                        dict(
                            type='ToTensor_OCV_Dict')],
                    training=True)),
            loaderConfITT=dict(
                type='PreloadedAssemble',
                ppConfs=[
                    dict(
                        type='allAreaScale',
                        eq=877.5),
                    dict(
                        type='allBaseCrop',
                        base=32),
                    dict(
                        type='allConvert2Dict'),
                    dict(
                        type='NormalizeRGB_OCV_Dict',
                        s=255),
                    dict(
                        type='ToTensor_OCV_Dict')])),
        optType='adam',
        flagUseLRS=True,
        lr=0.001,
        lrs=dict(
            type='FixedLength', 
            steps=[20000, 10000], lrs=[0.125e-3, 0.0625e-3, 0.01e-3]),
        trueValueGenerator=dict(
            type='OriScaleTrueValues', 
            trueDispMax=DISP_MAX, flagIntNearest=False),
        lossComputer=dict(
            type='OriScaleLoss',
            weights=[1., 1., 1., 1.], flagMaskedLoss=True, flagIntNearest=False),
        metric=dict(
            type='MetricKITTI'),
        testResultSubfolder='TestResults')
    )