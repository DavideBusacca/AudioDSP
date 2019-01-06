'''
 Copyright (C) 2018  Busacca Davide
 
 This file is part of PV.
 
 PV is free software: you can redistribute it and/or modify it under
 the terms of the GNU Affero General Public License as published by the Free
 Software Foundation (FSF), either version 3 of the License, or (at your
 option) any later version.
 
 PV is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 details.
 
 You should have received a copy of the Affero GNU General Public License
 version 3 along with PV.  If not, see http://www.gnu.org/licenses/
'''

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import utils as U

class Param_OverlapAdd(U.NewParam):
    def __init__(self, frameSize=4096, hopSize=2048):
        self.frameSize = frameSize
        self.hopSize = hopSize
    
class OverlapAdd(U.NewModule):
    def __init__(self, Param_OverlapAdd=None):
        self.setParam(Param_OverlapAdd)
        self.update()
        self.clear()
        
    def getParam(self):
        return Param_OverlapAdd(frameSize=self._frameSize_, hopSize=self._hopSize_)

    def getFrameSize(self):
        return self._frameSize_

    def getHopSize(self):
        return self._hopSize_

    def setParam(self, param_OverlapAdd=None):
        if param_OverlapAdd is None:
            param_OverlapAdd = Param_OverlapAdd()
        self.param_OverlapAdd = param_OverlapAdd
        
    def update(self):
        self._frameSize_ = self.param_OverlapAdd.frameSize
        self._hopSize_ = self.param_OverlapAdd.hopSize

        self._needsUpdate_ = False

    def process(self, x):
        if self._needsUpdate_ == True:
            self.update()

        nFrames = np.shape(x)[1]
        y = np.array(())
        for i in range(nFrames):
            y = np.append(y, self.clockProcess(x[:, i]))

        return y
        
    def clockProcess(self, x):
        self._y_[:self._frameSize_-self._hopSize_] = self._y_[self._hopSize_:]
        self._y_[-self._hopSize_:] = self._blankHop_
        self._y_ = self._y_ + x
        
        return self._y_[:self._hopSize_]
    
    def clear(self):
        self._y_ = np.zeros((self._frameSize_))
        self._blankHop_ = np.zeros((self._hopSize_))
        
def main():
    OLA = OverlapAdd(Param_OverlapAdd(hopSize=256, frameSize=1024))
    ones = np.ones((1024, 1))
    x = ones
    for i in range(10):
        x = np.hstack((x, ones))

    y = OLA.process(x)

    plt.plot(y)
    plt.show()
    
if __name__ == '__main__':
    main()