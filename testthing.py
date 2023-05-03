import chess.polyglot
# import matplotlib.pyplot as plot
import keras
import numpy
# from keras.optimizers import adamw_experimental
import numpy as np
import copy
import cProfile
import re
import time


#evalfunc2=copy.deepcopy(evalfunc)
#for z in range(0,len(evalfunc)):
    #evalfunc[z]=evalfunc2[z-z%64+8*(8-int((z%64)/8))+z%8-8]
#print(evalfunc)
j=[]
j.append([0,0,2])
j.append([0,5,1])
j=np.array(j)
o=(j.nonzero())

u=np.split(o[1],o[0])


print(u)
#u = u[u[:, 0].argsort()]

r=[0,8,8,3,2,2]
r=np.array(r)
o=[]
for w in u:
    o.append(r[w].sum())
#t=np.arange(1,u)

print(o)
