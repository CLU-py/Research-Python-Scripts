#import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%assign column names
cols = ['Height, km', 'O, cm-3', 'N2, cm-3', 'O2, cm-3', 'Mass_density, g/cm-3',
        'Temperature_neutral, k', 'He, cm-3', 'Ar, cm-3', 'H, cm-3', 'N, cm-3', 'F10_7_daily']

#read data file
path = '/import/SUPERDARN/matthew/misc/'
data = pd.read_csv(path + 'nrlmsise_82814919326.txt', sep = r'\s+' , skiprows = 38, header = None, names = cols)

#%%plot mass density profile
altitude = data['Height, km']
mass = data['Mass_density, g/cm-3']

fig = plt.subplots(figsize = (4, 8))
x = mass
y = altitude
plt.plot(x, y)

plt.xlabel('Mass Density (g/cm$^3$)')
plt.ylabel('Altitude (km)')
plt.xscale('log')
plt.tight_layout()
plt.savefig('mass_profile.png')

#%%plot temperature profile
temperature = data['Temperature_neutral, k']

fig = plt.subplots(figsize = (4, 8))
x = temperature
plt.plot(x, y)

plt.xlabel('Temperature (k)')
plt.ylabel('Altitude (km)')
plt.tight_layout()
plt.savefig('temperature_profile.png')

#annotate temperature profile with regions of the atmosphere (troposphere, stratosphere, mesosphere)