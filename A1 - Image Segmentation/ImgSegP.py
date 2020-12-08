# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt("skin.txt")
plt.imshow(A, cmap="gray")