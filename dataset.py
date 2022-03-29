import os
import numpy as np

#reads files
def readFile(fileLocation):
    with open(fileLocation, encoding='utf-8', errors='ignore', mode='r+') as file:
        data = file.read()
        data = data.replace("\n", "")                
        resetData =[]
        length = len(data)
        for index in range(length):                    
            resetData.append(float(data[index]))
    return resetData

# pulls features and labels form dataset
def pullDataset(path):
    datafile = os.listdir(path) 
    dataset=[]
    for file_name in datafile:
        names = file_name.split("_")
        fileContent = readFile(path + '/' + file_name)
        fileContent.append(int(names[1]))
        dataset.append(fileContent)
    dataset=np.asarray(dataset)
    x=dataset[:, :1024]
    y=dataset[:, 1024]
    x=np.asarray(x)    
    y =np.asarray(y)
    return x,y,dataset
