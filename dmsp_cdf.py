#import required modules
import cdflib
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from datetime import datetime

#%%set path to .cdf file
#cdf_file = cdflib.CDF('/home/mfw56/Downloads/dmsp-f17_ssj_precipitating-electrons-ions_20140321_v1.1.2.cdf') #path on wilcox.met
cdf_file = cdflib.CDF('E:\Python Scripts\Research\dmsp-f17_ssj_precipitating-electrons-ions_20140321_v1.1.2.cdf') #path on Matthew-PC

cdf_info = cdf_file.cdf_info()
#print(cdf_info.zVariables)

epoch_data = cdf_file.varget('Epoch') #epoch time; this is the x-axis
datetime_array = cdflib.cdfepoch.to_datetime(epoch_data) #convert CDF_EPOCH to datetime
datetime_series = pd.Series(datetime_array)
hhmm_array = datetime_series.dt.strftime('%H%M').values

aacgm_lat = cdf_file.varget('SC_AACGM_LAT') #geomagnetic latitude
#aacgm_lon = cdf_file.varget('SC_AACGM_LON') #geomagnetic longitude
aacgm_ltime = cdf_file.varget('SC_AACGM_LTIME') #MLT?

geocentric_r = cdf_file.varget('SC_GEOCENTRIC_R') #satellite altitude from the center of the earth in km

#eci = cdf_file.varget('SC_ECI')
#eci_label = cdf_file.varget('SC_ECI_LABEL')
channel_energies = cdf_file.varget('CHANNEL_ENERGIES') #range of energy channels; this is the y-axis

#electron_obs = cdf_file.varget('ELE_COUNTS_OBS') #observed electron count
#electron_bkg = cdf_file.varget('ELE_COUNTS_BKG') #background electron count
#electron_geometric = cdf_file.varget('ELE_GEOMETRIC')
#electron_average_energy = cdf_file.varget('ELE_AVG_ENERGY') #average electron energy

diff_energy_flux = cdf_file.varget('ELE_DIFF_ENERGY_FLUX') #electron flux per unit energy (erg/(cm^2 s sr eV)); this is the color or z-axis
#total_energy_flux = cdf_file.varget('ELE_TOTAL_ENERGY_FLUX') #electron flux over all measured energy channels (erg/(cm^2 s sr))

#%%set time interval based on start and end times
sttime = str(201403210400) #start time
ndtime = str(201403210600) #end time

sttime_dt = pd.to_datetime(sttime, format='%Y%m%d%H%M') #convert start time string to datetime object
ndtime_dt = pd.to_datetime(ndtime, format='%Y%m%d%H%M') #convert end time string to datetime object

interval = (datetime_series >= sttime_dt) & (datetime_series <= ndtime_dt) #set indicies that fall within the defined time interval

#size arrays according to the time interval
hhmm_array = hhmm_array[interval] # will be plotted on x-axis
aacgm_ltime = aacgm_ltime[interval] #will be plotted on x-axis
geocentric_r = geocentric_r[interval]

diff_energy_flux = diff_energy_flux[interval] #z-axis

#create time-energy spectrogram
fig, ax = plt.subplots(figsize = (10, 6))

x = np.arange(len(hhmm_array))

im = ax.pcolormesh(
    x, channel_energies, diff_energy_flux.T, #x, y, and z-axes
    cmap = 'jet', shading = 'nearest',
    vmin = np.min(diff_energy_flux), vmax = np.max(diff_energy_flux) #define color limits
    )

label_x = -10
label_y = -0.02

#add lables to UT axis
ax.set_xticks(np.linspace(0, len(x) - 1, 10))
ax.set_xticklabels(hhmm_array[ax.get_xticks().astype(int)])

ut_label = ax.xaxis.get_label()
#ax.set_xlabel('Time (UTC)')

#add satellite geomagnetic latitude to x-axis
ax2 = ax.secondary_xaxis('bottom')
ax2.set_xticks(ax.get_xticks())
ax2.set_xticklabels([f'{m:.1f}' for m in aacgm_lat[ax.get_xticks().astype(int)]])
ax2.spines['bottom'].set_position(('outward', 15)) #set the spine position
ax2.spines['bottom'].set_visible(False) #hide the spine of the second axis
ax2.tick_params(axis = 'x', which = 'both', length = 0) #hide tick lines

#add MLT to x-axis
ax3 = ax.secondary_xaxis('bottom')
ax3.set_xticks(ax.get_xticks())
ax3.set_xticklabels([f'{m:.1f}' for m in aacgm_ltime[ax.get_xticks().astype(int)]])
ax3.spines['bottom'].set_position(('outward', 30)) #set the spine position
ax3.spines['bottom'].set_visible(False) #hide the spine of the second axis
ax3.tick_params(axis = 'x', which = 'both', length = 0) #hide tick lines
#ax2.set_xlabel('MLT')

#add satellite altitude
ax3 = ax.secondary_xaxis('bottom')
ax3.set_xticks(ax.get_xticks())
ax3.set_xticklabels([f'{m:.1f}' for m in geocentric_r[ax.get_xticks().astype(int)]])
ax3.spines['bottom'].set_position(('outward', 45)) #set the spine position
ax3.spines['bottom'].set_visible(False) #hide the spine of the second axis
ax3.tick_params(axis = 'x', which = 'both', length=  0) #hide tick lines


ax.set_ylabel('Electron Energy (eV)')

cbar = fig.colorbar(im, ax = ax, orientation = 'vertical')
cbar.set_label('Differential Energy Flux')

#plt.tight_layout()



















