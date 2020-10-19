# =============================================================================
# Modified to backprop network
# =============================================================================
from backprop import backprop
import numpy as np
from read_data_set_into_array import get_data_sets
# Save a dictionary into a pickle file.
import pickle

def false_classification(a,Twos,notTwos):
    false_positive=0;
    classification=a.test(notTwos)
    false_positive = (classification == np.ones(classification.shape))*1
    total_misses=sum(false_positive);
    false_positive=sum(false_positive)*100/len(notTwos)  
    false_negative=0;
    classification=a.test(Twos)
    false_negative = (classification == np.zeros(classification.shape))*1
    total_misses+=sum(false_negative);
    false_negative=sum(false_negative)*100/len(Twos) #in precentage
    return false_negative,false_positive,total_misses

# --- part 2 - classify 2 and not 2 ---#

# =============================================================================
## uncomment this section if train_input_part2.p and test_input_park2.p doesnt exist in 
## parent directory
    
#train_input,test_input=get_data_sets() # every time i run this script, i reload the data set
#                                       # this takes forever. it will be best to pickle the data set out
#train=open( "train_input_part2.p", "wb" )
#pickle.dump( train_input, train )
#test=open( "test_input_part2.p", "wb" )
#pickle.dump( test_input, test )
#train.close()
#test.close()
# =============================================================================

train=open( "train_input_part2.p", "rb" )
test=open( "test_input_part2.p", "rb" )
train_input=pickle.load( train )
test_input=pickle.load( test ) 
train.close()
test.close()
                                     
train_itt_num=1000
 
#Targets:
target_sec2=np.array(()) 
for num in range(10):
    for example in range(250):       
        if num==0 and example==0:
            target_sec2=0;
        elif num != 2:
            target_sec2=np.vstack((target_sec2,np.array([0])))
        else:
            target_sec2=np.vstack((target_sec2,np.array([1])))

# lets see the false rates:
Twos=test_input[500:750,:]
notTwos=test_input[0:500,:];
notTwos=np.vstack((notTwos,test_input[750:2501,:]))

etta=0.5
train_itt_num=200
a=backprop(14**2,1,5)
# =============================================================================
##uncomment the following section if the the network has already been trained
##and weights of the trained network, pre saved at the parent directory
#a.train(train_input,target_sec2,train_itt_num,etta,'sigmoid')
#a.save('part2')
# =============================================================================
a.load('part2')
print('for {} perceptron itterations, and Batch algorithm (better than Online):'.format(train_itt_num))   
errors = (target_sec2!=a.test(test_input))*1
false_pos = ((target_sec2!=a.test(test_input))*1)*( target_sec2==1)
print('Total  Misses: %d/%d (%.2f%%)'%(sum(errors),len(errors),(100*sum(errors)/len(errors)))) 
print('The False Positive rate as Or calculates wrongly (see explanation in the code)\n See also that its equal to my False Negative rate/10:')
print('False Positive: %d/%d (%.2f%%)'%(sum(false_pos),len(false_pos),(100*sum(false_pos)/len(false_pos)))) 
print('The False Positive rate as should be:')
false_negative,false_positive,total_misses=false_classification(a,Twos,notTwos)
print('False Positive rate is: {:.2f}%'.format(false_positive[0]))
print('The False Negative rate as should be:')
print('False Negative rate is: {:.2f}%\n'.format(false_negative[0]))