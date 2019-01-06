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
    
class OverlapAdd(U.NewModule):
    def __init__(self, hopSize=512, frameSize=1024):
        self.hopSize = hopSize
        self.frameSize = frameSize
        self.clear()

        
    def getParam(self):
        pass
        
    def setParam(self):        
        #update OLA from ISTFT
        pass
        
    def update(self):
        pass

    def process(self):
        pass       
        
    def clockProcess(self, x):
        self._y_[:self.frameSize-self.hopSize] = self._y_[self.hopSize:]
        self._y_[-self.hopSize:] = self._blankHop_
        self._y_ = self._y_ + x
        
        return self._y_[:self.hopSize]
    
    def clear(self):
        self._y_ = np.zeros((self.frameSize))
        self._blankHop_ = np.zeros((self.hopSize))
        
def main(): 
    OLA = OverlapAdd(hopSize=256, frameSize=1024)
    x = np.ones((1024, 1))
    y = np.array(())
    for i in range(1600):
        y = np.append(y, OLA.clockProcess(x))

    plt.plot(y)
    plt.show()
    
if __name__ == '__main__':
    main()