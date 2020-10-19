from backprop import backprop
import numpy as np
from read_data_set_into_array import get_data_sets
import matplotlib.pyplot as plt
# Save a dictionary into a pickle file.
import pickle

# --- part 3 - classify 10 digits ---#

# =============================================================================
## uncomment this section if train_input_part3.p and test_input_part3.p doesnt exist in 
## parent directory

#train_input_sec5,test_input_sec5,target_sec5=get_data_sets('5') # every time i run this script, i reload the data set
#                                                                # this takes forever. it will be best to pickle the data set out
#train=open( "train_input_part3.p", "wb" )
#pickle.dump( train_input_sec5, train )
#test=open( "test_input_part3.p", "wb" )
#pickle.dump( test_input_sec5, test )
#target=open( "target_part3.p", "wb" )
#pickle.dump( target_sec5, target )
#train.close()
#test.close()
#target.close()
# =============================================================================
train=open( "train_input_part3.p", "rb" )
test=open( "test_input_part3.p", "rb" )
target=open( "target_part3.p", "rb" )
train_input_sec5=pickle.load( train )
test_input_sec5=pickle.load( test ) 
target_sec5=pickle.load( target ) 
train.close()
test.close()
target.close()



etta=0.5
train_itt_num=1000
a=backprop(14**2,10,16)
# =============================================================================
##uncomment the following section if the the network has already been trained
##and weights of the trained network, pre saved at the parent directory
a.train(train_input_sec5,target_sec5,train_itt_num,etta,'sigmoid','online',mu=0)
#a.save('part4')
# =============================================================================
a.load('part4')
errors = (np.argmax(target_sec5,axis=1)!=np.argmax(a.test(test_input_sec5),axis=1))*1
print('Total  Misses: %d/%d (%.2f%%)'%(sum(errors),len(errors),(100*sum(errors)/len(errors)))) 

#############confusion matrix##########
conf_mat=np.zeros((10,10))
ideal_mat=np.zeros((10,10))
for num in range(10):
    for i in range(250):
       ideal_mat[num , np.argmax(target_sec5[num*250+i,:])]+=1 # run the code with the target input and see we get diagonal matrix
       conf_mat[num , np.argmax(a.test(test_input_sec5)[num*250+i,:])]+=1
conf_mat=conf_mat*100/2500

plt.imshow(conf_mat, interpolation='nearest')
plt.show()
#print(conf_mat)
