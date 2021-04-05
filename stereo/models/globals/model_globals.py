
# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-20

FLAG_TF = (True, False, 1, 0)

PADDING_FIXED   = 'fixed'
PADDING_REFLECT = 'reflect'
PADDING_MODES   = ( PADDING_FIXED, PADDING_REFLECT )

class GLOBAL(object):
    TORCH_ALIGN_CORNERS = False
    TORCH_BATCH_NORMAL_TRACK = True
    TORCH_INSTANCE_NORMAL_TRACK = True
    TORCH_RELU_INPLACE = False
    PADDING_MODE = PADDING_REFLECT

    @staticmethod
    def torch_align_corners(flag=None):
        if ( flag is None ):
            return GLOBAL.TORCH_ALIGN_CORNERS
        else:
            assert( flag in FLAG_TF )
            GLOBAL.TORCH_ALIGN_CORNERS = flag

    @staticmethod
    def torch_batch_normal_track_stat(flag=None):
        if ( flag is None ):
            return GLOBAL.TORCH_BATCH_NORMAL_TRACK
        else:
            assert( flag in FLAG_TF )
            GLOBAL.TORCH_BATCH_NORMAL_TRACK = flag

    @staticmethod
    def torch_inst_normal_track_stat(flag=None):
        if ( flag is None ):
            return GLOBAL.TORCH_INSTANCE_NORMAL_TRACK
        else:
            assert( falg in FLAG_TF )
            GLOBAL.TORCH_INSTANCE_NORMAL_TRACK = flag

    @staticmethod
    def torch_relu_inplace(flag=None):
        if ( flag is None ):
            return GLOBAL.TORCH_RELU_INPLACE
        else:
            assert( flag in FLAG_TF )
            GLOBAL.TORCH_RELU_INPLACE = flag

    @staticmethod
    def padding_mode(mode=None):
        if ( mode is None ):
            return GLOBAL.PADDING_MODE
        else:
            assert( mode in PADDING_MODES ), f'Unexpected mode {mode}. Valid ones are {PADDING_MODES}'
            GLOBAL.PADDING_MODE = mode