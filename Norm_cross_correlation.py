import numpy as np

def norm_data(data):
    mean_data = np.mean(data)
    std_data = np.std(data,ddof=1)
    return (data-mean_data)/(std_data)

def ncc(data0,data1):
    return(1.0/(len(data0)-1))*np.sum(norm_data(data0)*norm_data(data1))



a = [1, 2, 3, 4]
b = [5, 3, 9, 8]

ncc1 = ncc(a,b)

print(ncc1)