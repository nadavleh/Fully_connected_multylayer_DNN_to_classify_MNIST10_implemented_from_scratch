from backprop import backprop
import numpy as np



# =============================================================================
# lets see the mumentum implementation on part 1
# =============================================================================
i=np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
targets=np.array([[0], [1], [1], [0]])

# with mumentum mu=0 the network requires 10k itterations
etta=0.5
a=backprop(2,1,3)
a.train(i,targets,10000,etta,'sigmoid','online',mu=0)
print('XOR gate classification with mu=0:')
print('input: [1 1] output: {}'.format(a.test(i,targets)[0]))
print('input: [0 1] output: {}'.format(a.test(i,targets)[1]))
print('input: [1 0] output: {}'.format(a.test(i,targets)[2]))
print('input: [0 0] output: {}'.format(a.test(i,targets)[3]))
# with mumentum mu=1 the network requires only 1k itterations
etta=15
a=backprop(2,1,3)
a.train(i,targets,1000,etta,'sigmoid','online',mu=1)
print('XOR gate classification with mu=1:')
print('input: [1 1] output: {}'.format(a.test(i,targets)[0]))
print('input: [0 1] output: {}'.format(a.test(i,targets)[1]))
print('input: [1 0] output: {}'.format(a.test(i,targets)[2]))
print('input: [0 0] output: {}'.format(a.test(i,targets)[3]))