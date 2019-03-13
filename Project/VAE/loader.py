import numpy as np


# Get the angles from the top500.txt file
def getData(fn):
    f = open(fn)
    data = []
    for line in f.readlines():
        if "#" in line:
            continue
        elif "NT" in line:
            continue
        elif "CT" in line:
            continue
        else:
            txt = line.split()
            phi = float(txt[2])
            psi = float(txt[3])
            angle = (phi, psi)
            data.append(angle)
    data = np.array(data)
    return data


#print(getData("top500.txt"))
