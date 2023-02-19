# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 00:04:15 2023

@author: Diloz
"""

import numpy as np


class data:
    def __init__(self, alphanum):
        self.alphanum = alphanum

    def tobinary(self):
        return ("".join([bin((ord(c)))[2:] for c in self.alphanum]))


class transmitter:
    def __init__(self, bits, pn):
        self.bits = bits
        self.pn = pn
        
        print("pn", pn)
    
    def spread(self):
        
        dataSpread = []
        for cnt in range(len(self.bits)):
            dig = np.int16(np.ones(len(self.pn)) * int(self.bits[cnt]))
            xor = np.bitwise_xor(dig, np.int16(self.pn))
            
            dataSpread.extend(xor)
            # dig = int(np.ones(len(self.pn))) * int(self.bits[cnt])
        
            print(self.bits[cnt])
            print(np.int16(self.pn))
            print(dig)
            print(xor)
            print(dataSpread)
            
            print(10*"-")

    

#%%
pseudocode = [1, 1, 0, 1, 1, 0, 0, 1, 1]
d1 = data("hi")

binar= d1.tobinary()

tx = transmitter(d1.tobinary(), pseudocode)
tx.spread()

