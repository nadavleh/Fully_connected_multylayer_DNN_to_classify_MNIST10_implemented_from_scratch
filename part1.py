from MultyLayerBackProp import backprop
import numpy as np

# --- testing perceptron ---#
#AND gate        
i=np.array([[1, 1],
            [0, 1],
            [1, 0],
            [0, 0]])
targets=np.array([[1.],
                  [0.],
                  [0.],
                  [0.]])
# =============================================================================
# implement Batch and Online algorithms and see that both work well
# =============================================================================
etta=0.5
####################AND gate############################
#a=backprop(2,1,3)
#a.train(i,targets,5000,etta)
#print('AND gate classification:')
#print('Batch Algorithm')
#print('input: [1 1] output: {}'.format(a.test(i,targets)[0]))
#print('input: [0 1] output: {}'.format(a.test(i,targets)[1]))
#print('input: [1 0] output: {}'.format(a.test(i,targets)[2]))
#print('input: [0 0] output: {}'.format(a.test(i,targets)[3]))


#####################AND gate & OR gate#####################
# i.e. perceptron with two inputs and two outputs 
#targets=np.hstack((targets,np.ones((len(targets),1))))
#targets[3,1]=0
#a=backprop(2,2,2)
#a.train(i,targets,10000,etta)
#print('AND & OR gate classification:')
#print('input: [1 1] output: {}'.format(a.test(i,targets)[0]))
#print('input: [0 1] output: {}'.format(a.test(i,targets)[1]))
#print('input: [1 0] output: {}'.format(a.test(i,targets)[2]))
#print('input: [0 0] output: {}'.format(a.test(i,targets)[3]))
####################XOR gate############################
targets=np.array([[0.],
                  [1.],
                  [1.],
                  [0.]])

h=np.array([2,3,3,1])
a=backprop(h)
a.train(i,targets,10000,etta,'sigmoid')
print('XOR gate classification:')
print('input: [1 1] output: {}'.format(a.test(i,targets)[0]))
print('input: [0 1] output: {}'.format(a.test(i,targets)[1]))
print('input: [1 0] output: {}'.format(a.test(i,targets)[2]))
print('input: [0 0] output: {}'.format(a.test(i,targets)[3]))
