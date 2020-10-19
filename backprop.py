import numpy as np
# Save a dictionary into a pickle file.
import pickle

class backprop():
    def __init__(self,n=2,m=1,h=1,sigma=0.05): 
        # n is number of inputs, h hidden layer and m is outputs        
        self.input_len=n
        self.output_len=m
        self.hidden=h
        self.w_ho = sigma*np.random.randn(m,h+1)
        self.w_ih = sigma*np.random.randn(h,n+1)

    def __str__(self):
        return '3 layer backprop network with {} inputs and {} outputs, hidden layer is with {} nodes  '.format(self.input_len,self.output_len,self.hidden)
    
    def test(self, input, target=[], show='no'):
        # "input" is a matrix where each row is a different input vector, so we append 1's for bias a collumn on the right
        input=np.hstack( (input, np.ones((input.shape[0],1)) ) )
        # "hidden" is a matrix of hidden layer outputs, each row correspond to each row in the input vector
        # this is possible only if we matrix multiply input*(w_ih)^T  
        hidden=np.matmul(input,self.w_ih.T)
        # apply the sigmoid function to the "hidden" matix
        hidden=f(hidden)
        # append 1's collumn on the right for biasses
        hidden=np.hstack( (hidden, np.ones((hidden.shape[0],1)) ) )
        # get output layer as matrix where each row correspondsto each row in the input vector
        output=np.matmul(hidden,self.w_ho.T)
        # apply the sigmoid function to the "output" matix (the backprop train algorithm is based of this so we 
        # cant get away with just a step function like in the perceptron because its non differentiable)
        output=f(output)
#        print(output)
        
        # if we have m=1 i.e. one output, then "output" is a collumn vector with number of rows like "input" matrix 
        # (just input.shape[0] number of patterns/examples). we thus get an output which may look like [0.2 0.3 -0.2] (as
        # a collumn vec) so what do we need to do? the classification is not a step function but the continous sigmoid.
        # to we now do a step function? the classification cant remain a float! this doesnt make sense both theoretically 
        # as the backprop derrivation relies on differentiable func's and empirically as can be seen in the AND  gate example
        if len(output[0,:])==1:
#            return output
            output=(output>0.7)*1
        # if we have multiple m outputs then the i think the highest number in each row, is the most likely classification,
        # thus i replace it with a 1 and zero all the other ellements in the row.
        else:
            idx=np.argmax(output,1) #idx is an array of the index if maximum value in each row
            output=np.zeros(output.shape)
            for i in range(output.shape[0]):
                output[i,idx[i]]=1

        return output
        
    def train(self, training_inputs, targets ,num_of_itter=1000, etta=0.5, act_fun = 'sigmoid',method='online',mu=0):
        num_of_patters=len(targets) # number of training examples is based upon 
                                    # the number of targets
                                   
        # "training_inputs" is a matrix where each row is a different input vector, so we append 1's for bias a collumn on the right          
        training_inputs=np.hstack( (training_inputs, np.ones((len(training_inputs),1)) ) ) 
        # loop through user specified number of itterations. 
        # This is the algorithm for backprop just as seen in the lectures
        if method == 'online':
            for i in range(num_of_itter):
                if i%100==0:
                    print('itteration: {}/{}'.format(i,num_of_itter))
                dw_ih = np.zeros(self.w_ih.shape)
                dw_ho = np.zeros(self.w_ho.shape)
                dw_ih_last=dw_ih
                dw_ho_last=dw_ho
                for j in range(num_of_patters):
                    # "H_net" (H for hidden) is a vector of hidden layer outputs, yeilded by a matrix
                    # multiplication of w_ih*training_inputs[j,:], where training_inputs[j,:] is the j'th pattern/example with n inputs
                    H_net=np.dot(self.w_ih,training_inputs[j,:])
                    # apply the sigmoid function to the "hidden" ector
                    H=f(H_net,act_fun)                 
                    # append 1 to the vector for the bias weight
                    H=np.append(H,1)                
                    # get output layer vector, yeilded by a matrix multiplication of w_ho*H
                    O_net=np.dot(self.w_ho,H)
                    # apply the sigmoid function to the output vector
                    O=f(O_net,act_fun)
                    # delta_O is a vector of m entries. thus, because (targets[j,:]-O) is also m entries long
                    # we perform ellement wise multiplication with f_prime(O_net) (which is m entries long)
                    # (the algorithm in the slides doesnt specifically state ellement wise multiplication and thats a shame
                    # because the derrivation says it must be so, and so does the delta_O dimenstion analysis. there's no other posibility)
                    delta_O=(targets[j,:]-O)*f_prime(O_net,act_fun);                                               
                    # again, the slides algorithm states delta_H=\delta_O*(w_ho^T)*f_prime(H_net) which is simply not clear how to implement theorretically wise and
                    # dimention wise (there is no clear difference in the slides between ellement wise multiplication and matrix multiplication)
                    # explenation: \delta_O*(w_ho^T) is actually a matrix multiplication of (w_ho^T)*\delta_O. this is true from the backprop derrivation
                    # and also the dimensions add up: w_ho is (m)x(h+1) and so w_ho^T is (h+1)x(m), where \delta_O is (m)x(1).
                    # we dont compute delta_H in a single step because we need to discard of the last ellement (the last ellement is meaningless becase its
                    # the bias weights row scallarly multiplied with delta_O). this is because f_prime(H_net) is an (h)x(1) vector and we need to
                    # ellement wise multiply it with (w_ho^T)*\delta_O
                    delta_OO=np.dot(self.w_ho.T,delta_O)
                    delta_OO=delta_OO[0:len(delta_OO)-1] #discard last ellement which is meaningless
                    delta_H=delta_OO*f_prime(H_net,act_fun)               
                    # once again, the slides are unclear with the type of opperation we need to perform. we now adjust each weight
                    # by a \Delta(w) proportional to the input. this is just like the Gradient descent implies (which justifies the perceptron approach aswell)
                    # and just as we derived in the slides.
                    # dimensions: training_inputs[j,:] is (n+1)x(1), delta_H is (h)X(1) and so an outer product training_inputs*delta_H^T is
                    # (n+1)x(h). because self.w_ih (h)x(n+1) we need to transpose to get fitting dimensions
                    dw_ih+=np.outer(training_inputs[j,:],delta_H).T
                    dw_ho+=np.outer(H,delta_O).T
                    self.w_ih += etta*dw_ih+mu*dw_ih_last
                    self.w_ho += etta*dw_ho+mu*dw_ho_last
                    dw_ih_last=dw_ih #momentum
                    dw_ho_last=dw_ho #momentum
                    dw_ih = np.zeros(self.w_ih.shape)
                    dw_ho = np.zeros(self.w_ho.shape)

        if method == 'batch':
            for i in range(num_of_itter):
                if i%100==0:
                    print('itteration: {}/{}'.format(i,num_of_itter))
                dw_ih = np.zeros(self.w_ih.shape)
                dw_ho = np.zeros(self.w_ho.shape)          
                for j in range(num_of_patters):
                    H_net=np.dot(self.w_ih,training_inputs[j,:])
                    H=f(H_net,act_fun)                 
                    H=np.append(H,1)                
                    O_net=np.dot(self.w_ho,H)
                    O=f(O_net,act_fun)
                    delta_O=(targets[j,:]-O)*f_prime(O_net,act_fun);                                               
                    delta_OO=np.dot(self.w_ho.T,delta_O)
                    delta_OO=delta_OO[0:len(delta_OO)-1] #discard last ellement which is meaningless
                    delta_H=delta_OO*f_prime(H_net,act_fun)               
                    dw_ih+=np.outer(training_inputs[j,:],delta_H).T
                    dw_ho+=np.outer(H,delta_O).T                 
                self.w_ih += etta*dw_ih/num_of_patters
                self.w_ho += etta*dw_ho/num_of_patters
    def save(self, part = ''):
        outputing1=open( "weights"+"_"+part+"_IH.p", "wb" )
        pickle.dump( self.w_ih, outputing1 )
        outputing2=open( "weights"+"_"+part+"_HO.p", "wb" )
        pickle.dump( self.w_ho, outputing2 )
        outputing1.close()
        outputing2.close()
    def load(self, part = ''):
        inputing1=open( "weights"+"_"+part+"_IH.p", "rb" )
        inputing2=open( "weights"+"_"+part+"_HO.p", "rb" )
        self.w_ih=pickle.load( inputing1 )
        self.w_ho=pickle.load( inputing2 )
        inputing1.close()
        inputing2.close()
#        print(self.w_ih)
        
        
        
def f(x,method='sigmoid'):
    if method=='sigmoid':
        return 1 / (1 + np.exp(-x))
    if method == 'tanh':
        return 0.5*np.tanh(x)+1
    if method == 'ReLU':
        return np.maximum(0,x)
    else:
        return 0;

def f_prime(x,method='sigmoid'):
    if method=='sigmoid':
        return f(x,'sigmoid')*(np.ones(len(x)) - f(x,'sigmoid'))
    if method == 'tanh':
        return 0.5*np.ones(len(x)) - 0.5*f(x,'tanh')**2
    if method == 'ReLU':
        return (x>0)*1
    else:
        return 0;
    
                