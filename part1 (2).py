from backprop import backprop
import numpy as np

#####################XOR gate############################
      
i=np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
targets=np.array([[0], [1], [1], [0]])
etta=0.5
a=backprop(2,1,3)
a.train(i,targets,10000,etta,'sigmoid')
print('XOR gate classification:')
print('input: [1 1] output: {}'.format(a.test(i,targets)[0]))
print('input: [0 1] output: {}'.format(a.test(i,targets)[1]))
print('input: [1 0] output: {}'.format(a.test(i,targets)[2]))
print('input: [0 0] output: {}'.format(a.test(i,targets)[3]))
