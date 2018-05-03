import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.figure()
for f in ['results_approx_1525319014.p', 'results_approx_1525319022.p', 'results_approx_1525319248.p', 'results_approx_1525319552.p']:
    print (f)
    curr  = pickle.load( open( 'results/'+f, "rb" ) )
    label = 'bins:'+str(curr['n_bins'])+', gamma:'+str(curr['gamma'])+ ', eta:'+str(curr['eta'])+'; decay:'+str(curr['epsilon_decay'])
    plt.plot(curr['hist'], label=label)
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend()
plt.tight_layout()
plt.savefig('approx_fig.png')
