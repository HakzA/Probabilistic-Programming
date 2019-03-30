import os
import numpy as np
from random import shuffle
# Pickle is used for saving/dumping and loading Python obecjts into a file
import six.moves.cPickle as pickle


# Get the angles from the top500.txt file
def getData(base_path, fn):
    print("Getting the dataset ready...")
    # Making the path for the proteins.pkl file (we will store the dataset here)
    output = os.path.join(base_path, fn)
    # Open the original protein dataset for reading
    f = open('top500.txt', 'r')
    # We will go through the proteins in the file and store them in this list
    data = []
    for line in f.readlines():
        if "#" in line:
            continue
        elif "NT" in line:
            continue
        elif "CT" in line:
            continue
        else:
            # Split text file into:
            # ['Amino acid symbol', 'Secondary structure symbol', 'Phi-angle', 'Psi-angle']
            txt = line.split()
            # Store the phi-angle
            phi = float(txt[2])
            # Store the psi-angle
            psi = float(txt[3])
            # Combine the angles in a tuple
            angle = (phi, psi)
            # Append that tuple onto the data array
            data.append(angle)

    # Convert from list to numpy array
    data = np.array(data)

    # Shuffle dataset so avoid bias
    # (don't know if we need this, since we don't compute for sequences like TorusDMM)
    shuffle(data)

    # We want to split our dataset into two datasets:
    # Train dataset: used for training (70% of the data)
    # Test dataset: used for testing (30% of the data)
    p_70 = np.int(0.7*len(data))
    # Train dataset = data[0]
    # Test dataset = data[1]
    data = [data[:p_70], data[p_70:]]
    pickle.dump(data, open(output, "wb"), pickle.HIGHEST_PROTOCOL)


    #return data

# Initiated upon import, so the proteins.pkl is created from the beginning
base_path = './data'
getData(base_path, "proteins.pkl")
proteins_path = os.path.join(base_path, "proteins.pkl")


#_________HELPER FUNCTIONS FOR PICKLE FILE_________

# Load proteins.pkl from disk
def load_data():
    with open(proteins_path, "rb") as f:
        return pickle.load(f)







#print(getData("top500.txt"))
