#!/usr/bin/env python3

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
#plt.rcParams.update({'text.usetex':True})

#plt.set_loglevel('debug')

plt.plot([1, 2, 3])
plt.xlabel(r'$\alphas')
plt.savefig('myplot.pdf')
print('Plot has been made')
