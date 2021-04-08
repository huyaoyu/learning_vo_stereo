# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-04-08

import numpy as np

from .register import ( VISUALIZERS, make_visualizer )

class FigureGrid(object):
    def __init__(self, rows, cols, cellShape, cells):
        '''
The cells will look like the following:
cells=[
    dict(type=<visualizer type>, idx=<2-element, h w>, <other keywords>),
    dict(type=<visualizer type>, idx=<2-element, h w>, <other keywords>),
]'''
        super(FigureGrid, self).__init__()

        self.rows = rows
        self.cols = cols
        self.cellShape = cellShape

        # self.visualizers is a dict. The keys are
        # created as '<idx[0]>x<idx[1]>'.
        # Each visualizer is a dict with keys
        # 'visualizer', 'idx', 'referenced'.
        # Where 'referenced' is a list of 2-element, 
        # Showing if any other cells are referencing 
        # this cell.
        self.visualizers = dict()
        for conf in cells:
            ir, ic = conf.pop('idx')
            assert ( 0 <= ir < self.rows ), f'ir = {ir}, self.rows = {self.rows}. '
            assert ( 0 <= ic < self.cols ), f'ic = {ic}, self.cols = {self.cols}. '

            if ( conf['type'] == 'Reference' ):
                rr, rc = conf.pop('refIdx')
                assert ( 0 <= rr < self.rows ), f'rr = {rr}, self.rows = {self.rows}. '
                assert ( 0 <= rc < self.cols ), f'rc = {rc}, self.cols = {self.cols}. '
                # Compose the key string.
                key = '%dx%d' % ( rr, rc )

                # Test if this visualizer is already defined.
                if ( key in self.visualizers ):
                    # Update the 'referenced' value.
                    self.visualizers[key]['referenced'].append( [ir, ic] )
                else:
                    # Create a partial visualizer definition.
                    self.visualizers[key] = dict(referenced=[ [ir, ic] ])
            else:
                # Compose the key string.
                key = '%dx%d' % ( ir, ic )
                conf['newShape'] = self.cellShape

                # Test if this visualizer is already defined.
                if ( key in self.visualizers ):
                    # Update the partially defined visualizer.
                    self.visualizers[key]['idx'] = [ ir, ic ]
                    self.visualizers[key]['visualizer'] = make_visualizer( VISUALIZERS, conf )
                else:
                    # Create new visualizer.
                    self.visualizers[key] = dict( 
                        idx=[ir, ic], 
                        visualizer=make_visualizer( VISUALIZERS, conf ),
                        referenced=[] )

    def __call__(self, ctx, batchIdx):
        H, W = self.cellShape[:2]

        # The canvas.
        canvas = np.zeros( 
            ( self.rows * self.cellShape[0], self.cols * self.cellShape[1], 3 ), 
            dtype=np.uint8 ) + 255

        for _, visDict in self.visualizers.items():
            # Create the figure.
            img = visDict['visualizer'](ctx, batchIdx)

            # Insert the figure into position.
            rIdx, cIdx = visDict['idx']
            canvas[ rIdx*H:(rIdx+1)*H, cIdx*W:(cIdx+1)*W, ... ] = img

            # Check if any other cell is referencing this cell.
            for indices in visDict['referenced']:
                rIdx, cIdx = indices
                canvas[ rIdx*H:(rIdx+1)*H, cIdx*W:(cIdx+1)*W, ... ] = img

        return canvas