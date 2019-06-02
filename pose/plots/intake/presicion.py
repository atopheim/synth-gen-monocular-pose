from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-poster')
style.use('ggplot')
import csv
import numpy as np

x_train = []
y_train = []

x_val = []
y_val = []

x_occ = []
y_occ = []

def sliding_mean(data_array, window=5):  
    data_array = data_array  
    new_list = []  
    for i in range(len(data_array)):  
        indices = range(max(i - window + 1, 0),  
                        min(i + window + 1, len(data_array)))  
        avg = 0  
        for j in indices:  
            avg += data_array[j]  
        avg /= float(len(indices))  
        new_list.append(avg)  
          
    return new_list


with open('/home/volvomlp2/python-envs/pvnet/data/record/intake_homemade_train.log','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for idx, row in enumerate(plots):
        if idx%10 == 0:
            x_val.append(int(row[2])) # num epoch
            y_val.append(float(row[12])) # metric
        else:
            x_train.append(int(row[2]))
            y_train.append(float(row[12]))

# 6 - segmentation , 9 - vertex, 12 - precision, 15 - recall 

#array creation
x_train = np.array(x_train)
y_train = np.array(y_train)
print("y_train",y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
y_train_smooth = sliding_mean(y_train)
y_val_smooth = sliding_mean(y_val,10)
y_val_smooth = sliding_mean(y_val_smooth)
#y_val_smooth = sliding_mean(y_val_smooth)
plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")  
#plt.plot(x_train,y_train_gauss, 'r',label='Training metrics')
#plt.plot(x_val,y_val_gauss, 'b', label='Validation metrics')


plt.plot(x_train,y_train_smooth, 'r', label = 'Training metric')
plt.plot(x_val,y_val_smooth,'g',label= 'Validation metric')

#plt.plot(x_occ,y_occ, 'g', label='Occlusion metrics')
plt.xlabel('epochs')
plt.ylabel('precision')
plt.title('Precision\nObject - INTAKE')
plt.legend()
plt.show()

