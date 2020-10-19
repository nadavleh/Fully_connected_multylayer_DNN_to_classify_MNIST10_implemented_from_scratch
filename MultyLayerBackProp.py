import numpy as np

class backprop():
    def __init__(self,h,sigma=0.05): 
        # n is the number of inputs, m is number of outputs, h_i is the number of nodes in the i^th hidden layer
        # and so the input to the constructor is a full specification of the networks demensions:
        # h = [n,h_1,...,h_i,...,m]        
        self.input_len=h[0]
        self.output_len=h[len(h)-1]
        self.h=h;
        self.hidden=h[1:len(h)]
        self.HLnum=len(self.hidden)
        #initialize network matrices in the form of a list
        self.W=[];
        self.dW=[];
        for i in range(len(self.h)-1):
          self.W.append( sigma*np.random.randn(self.h[i+1],self.h[i]+1) ) 
          self.dW.append( np.zeros([self.h[i+1],self.h[i]+1]) )


    def __str__(self):
        return '{} hidden layers backprop network with {} inputs and {} outputs, hidden layer is with {} nodes  '.format(self.HLnum,self.input_len,self.output_len,self.hidden)
    
    def test(self, input, target=[], show='no'):
        input=np.hstack( (input, np.ones((len(input),1)) ) ) #append ones to each pattern, for bias  
#        print('input are: ',input )
        input=input.T# move convention of each pattern as column vector so as to agree with the algorithm i wrote by hand
#        print('input.T are: ',input )
        num_of_patters=input.shape[1]
        output=np.zeros([num_of_patters,self.output_len])
         
        # the pattern loop can be replaced with matrix operations but fuck that shit
        for p in range(num_of_patters):
            #Forward propogate the input
            z=[input[:,p]];
            y=[input[:,p]];
            for j in range(len(self.h)-1):
                z.append( np.matmul(self.W[j],y[j]) )               
                y.append(np.append([f(z[j+1])],[1]))
            y[-1]=np.delete(y[-1],y[-1][-1]) #delete last ellement of the last layer activation
            output[p,:]=y[-1];
        
        if len(output[0,:])==1:
            return output
#            output=(output>0.7)*1
        # if we have multiple m outputs then the i think the highest number in each row, is the most likely classification,
        # thus i replace it with a 1 and zero all the other ellements in the row.
        else:
            idx=np.argmax(output,1) #idx is an array of the index if maximum value in each row
            output=np.zeros(output.shape)
            for i in range(output.shape[0]):
                output[i,idx[i]]=1

        return output
        
    def train(self, training_inputs, targets ,num_of_itter=1000, etta=0.5, act_fun = 'sigmoid'):
        
        training_inputs=np.hstack( (training_inputs, np.ones((len(training_inputs),1)) ) ) #append ones to each pattern, for bias
        training_inputs=training_inputs.T # move convention of each pattern as column vector so as to agree with the algorithm i wrote by hand
        targets=targets.T
        
        
        num_of_patters=targets.shape[1] 
        
         
        for i in range(num_of_itter):
            if i%100==0:
                print('itteration: {}/{}'.format(i,num_of_itter))        
            for p in range(num_of_patters):
                #Forward propogate the input
                z=[training_inputs[:,p]];
                y=[training_inputs[:,p]];
                for j in range(len(self.h)-1):
                    z.append( np.matmul(self.W[j],y[j]) )
                    
                    y.append(np.append([f(z[j+1],act_fun)],[1]))
                y[-1]=np.delete(y[-1],y[-1][-1]) #delete last ellement of the last layer activation because
                                                 #its the output and we dont need to append a bias
   
                
                #backward propogate the error
                for j in range(len(self.h)-1,0,-1): #does it include 0?
                    if j==(len(self.h)-1):
                        dEdy_next =  -(targets[:,p] - y[j])#*f_prime(z[j])
                        dEdy=dEdy_next
                    else:
                        dEdy=np.matmul(self.W[j].T,(dEdy_next*f_prime(z[j+1],act_fun) ))
                        dEdy = dEdy[0:len(dEdy)-1] #discard last ellement
                        dEdy_next = dEdy
                    self.dW[j-1]+=-etta*np.outer( (dEdy*f_prime(z[j],act_fun)) , y[j-1] )

                
                #update the weights in each patern 'online' style
                for j in range(len(self.h)-1):
                    self.W[j]+=self.dW[j]
                    self.dW[j]=np.zeros(self.dW[j].shape)
                
#            #update the weights in each patern 'batch' style
#            for j in range(len(self.h)-1):
##                print(self.dW[j]/num_of_patters)
#                self.W[j]+=self.dW[j]/num_of_patters
#                self.dW[j]=np.zeros(self.dW[j].shape)
# 

             
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
    
                