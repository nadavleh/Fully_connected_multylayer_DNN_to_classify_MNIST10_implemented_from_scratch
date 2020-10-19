# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:54:49 2019

@author: User
"""
import numpy as np
from matplotlib import pyplot as plt



def extract_num(num, example,lines, imshow='no'):
    from_row=num*250*16+example*16+1
    #to_row=num*250*16+example*16+15
    for i in range(14):
        if i==0:
            num_array= np.fromstring(lines[from_row], dtype=float, sep=' ');  
        else:
            num_array=np.vstack((num_array,np.fromstring(lines[from_row+i], dtype=float, sep=' ')))
    if imshow=='show':
        plt.imshow(num_array, interpolation='nearest')
        plt.show()
    return num_array;

def num_class(num):
    arr=np.zeros(10)
    arr[num]=1;
    return arr
def arrange_as_row(array):
    array=np.reshape(array, len(array)**2)
    return array;

def get_data_sets(section='not_section5'):
    # same script as part2 but as a function, so can be called in part 3
    f=open('digits_train.txt')
    lines=f.readlines()
    train_input=np.array(())
    target_sec5=np.array(())
    for num in range(10):
        for example in range(250):
            some_number=extract_num(num, example, lines)
            some_number=arrange_as_row(some_number)
            some_number_class=num_class(num)
            if num==0 and example==0:
                train_input=some_number;
                target_sec5=some_number_class;
            else:
                train_input=np.vstack((train_input,some_number))
                target_sec5=np.vstack((target_sec5,some_number_class))
    f=open('digits_test.txt')
    lines=f.readlines()
    test_input=np.array(())
    for num in range(10):
        for example in range(250):
            some_number=extract_num(num, example, lines)
            some_number=arrange_as_row(some_number)
            some_number_class=num_class(num)
            if num==0 and example==0:
                test_input=some_number;           
            else:
                test_input=np.vstack((test_input,some_number))
   # return train_input, test_input target_sec5 if (section == '5') 
    if section == '5':
        return train_input, test_input, target_sec5
    else:
        return train_input, test_input
