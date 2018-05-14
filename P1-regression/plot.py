import matplotlib.pyplot as plt
import numpy as np
import csv

x_axis = [2 * (i) for i in range(12)]
loss = []
val_loss = []

with open('nnloss.csv') as f:
    reader = csv.reader(f, delimiter=',')

    next(reader, None)

    for i, row in enumerate(reader):
        loss.append(np.sqrt(float(row[0])))
        val_loss.append(np.sqrt(float(row[1])))

plt.figure()
plt.xticks(x_axis)
plt.plot(range(len(loss)), loss, label='loss')
plt.plot(range(len(val_loss)), val_loss, label='val_loss')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend()
# plt.show()
plt.savefig('plot.png')
