from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, Descriptors, AllChem
from rdkit import DataStructs
import numpy as np
import gzip
import io

f1 = gzip.open('test.csv.gz', 'rb')
next(f1, None)

fbuf = io.BufferedReader(f1)

X_test = np.zeros((200000, 2048))
# X_train = np.zeros((824230, 2048))
# y_train = np.zeros((824230))

for j, line in enumerate(fbuf):
    if j == 200000:
        continue
    elif j == 400000:
        break

    line = line.decode('utf-8').split(',')
    smile = line[1]

    l = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(l, arr)
    X_test[j - 200000] = arr

    if j % 2048 == 0:
        print(j)

np.save('X_test_200-400K.npy', arr)
