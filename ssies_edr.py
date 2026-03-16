#!/usr/bin/env python3

#import required modules
import re
import glob
import gzip
import aacgm
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from datetime import datetime, timedelta

#%%define functions
#define function for parsing the ephemeris
def parse_ephemeris(line):
    lat_geo = float(line[0:9])
    lon_geo = float(line[10:19])
    lat_apex = float(line[20:29])
    lon_apex = float(line[30:39])

    apex_lt = float(line[40:53])
    alt_km = float(line[54:62])
    
    return [lat_geo,
            lon_geo,
            lat_apex,
            lon_apex,
            apex_lt,
            alt_km]

#define function for parsing satellite potential
def parse_potential(line, nvals):
    values = []
    
    width = 12
    step = 13 #12 characters plus 1 space
    
    for i in range(nvals):
        start = i * step
        end = start + width
        val_str = line[start:end].strip()
        val = float(val_str)
        
        if val == -0.10000E+38:
            val = np.nan
        
        values.append(val)
        
    return values

#define function for parsing blocks with the E12.5 format (satellite potential, primary plasma density, vertical/horizontal ion drift)
def parse_e12(line, nvals):
    values = []
    
    width = 12
    step = 13 #12 characters plus 1 space
    
    for i in range(nvals):
        start = i * step
        end = start + width
        val_str = line[start:end].strip()
        val = float(val_str)
        
        if val == -0.10000E+38:
            val = np.nan

        values.append(val)
        
    return values

#define function for getting tick labels
def get_labels(tick_locs, data_array, dataframe):
    tick_datetimes = mdates.num2date(tick_locs)
    labels = []
    for t in tick_datetimes:
        target = t.replace(tzinfo = None)
        idx = dataframe.index.get_indexer([target], method = 'nearest')[0]
        val = data_array[idx]
        labels.append(f"{val:.1f}" if not np.isnan(val) else "")
    return labels

#%%set start date for reading specific .gz file
sttime = str(201403210900)
ndtime = str(201403211030)
date_str = sttime[0:8] #year, month, and day string

sat = 'f16' #set satellites from f15 to f18
sat_num = sat[1:3]

data_path = '/import/SUPERDARN/matthew/dmsp/ssies_edr/' + sttime[0:8] + '/'
pattern = re.compile(rf"SIES\d-{sat}.*?\.GLOBAL_DD\.{date_str}_TP\.\d{{6}}-\d{{6}}_DF\.EDR\.gz$")

files = glob.glob(data_path + '*.gz') #finds all .gz files in the specified data path
for file in files:
    if sat_num in file: #find the file for the current satellite
        fname = file
        
#%%open .gz file
blocks = []
block_size = 114

ephemeris = []
sat_potential = [] #the electric potential of the spacecraft relative to the surrounding plasma
primary_density = []
horizontal_drift = []
vertical_drift = []

with gzip.open(fname, 'rt') as f:
    lines = [line.rstrip("\n") for line in f] #read the lines of the record and keep all blank lines
    
    #parse the minute blocks
    for i in range(0, len(lines), block_size):
        block = lines[i:i + block_size]
        
        if len(block) < block_size:
            continue
        
        blocks.append(block)
        
        #get the time for each block
        metadata = block[2]
        parts = metadata.split()
        date = parts[3]
        time = parts[4]
        
        dt = datetime.strptime(date + time, '%Y%m%d%H%M') #convert the metadata to a datetime object
        
        #======================================================================
        #get and append ephemeris data for each block
        time_ephemeris = np.arange(0, 60, 20)
        time_ephemeris = time_ephemeris.tolist() #seconds each minute for ephemeris measurements
        
        for i, idx in enumerate([4,5,6]):
            try:
                timestamp = dt + timedelta(seconds = time_ephemeris[i])
                parsed = parse_ephemeris(block[idx])
                ephemeris.append([timestamp] + parsed)

            except:
                continue
            
        #======================================================================
        #get and append satellite potential for each block
        line1 = block[8]
        line2 = block[9]
        
        vals1 = parse_potential(line1, 8)
        vals2 = parse_potential(line2, 7)
        vals_potential = vals1 + vals2 #satellite potential values for the current block (in volts)
        
        #source flag for satellite potential (last 2 characters)
        source_potential = int(line2[-2:].strip())
        if source_potential == 1:
            source_potential_str = 'on-board microprocessor'
            
        if source_potential == 2:
            source_potential_str = 'SENPOT sensor'
        
        time_potential = np.arange(0, 60, 4)
        time_potential = time_potential.tolist() #seconds each minute for satellite potential
        
        for i, val in enumerate(vals_potential):
            try:
                timestamp = dt + timedelta(seconds = time_potential[i])
                sat_potential.append([timestamp, val, source_potential])
            
            except:
                continue
            
        #======================================================================
        #get and append primary plasma density for each block
        vals_primary_density = [] #primary plasma density values for the current block (in #/cm^3)
        
        time_minute = np.arange(0, 60, 1)
        time_minute = time_minute.tolist() #seconds each minute for many measurements
        
        for idx in range(11, 21):
            line = block[idx]
            
            vals = parse_e12(line, 6)
            vals_primary_density.extend(vals)
            
        #source flag
        source_primary_density = int(block[21].strip())
        
        for i, val in enumerate(vals_primary_density):
            try:
                timestamp = dt + timedelta(seconds = time_minute[i])
                primary_density.append([timestamp, val, source_primary_density])
                
            except:
                continue
            
        #======================================================================
        #get and append horizontal ion drift velocities for each block
        vals_hori_drift = [] #horizontal ion drift velocities for the current block (in m/s)
        
        for idx in range(23, 33):
            line = block[idx]
            
            vals = parse_e12(line, 6)
            vals_hori_drift.extend(vals)
            
        for i, val in enumerate(vals_hori_drift):
            try:
                timestamp = dt + timedelta(seconds = time_minute[i])
                horizontal_drift.append([timestamp, val])
                
            except:
                continue
                
        #======================================================================
        #get and append vertical ion drift velocities for each block
        vals_vert_drift = [] #vertical ion drift velocities for the current block (in m/s)
        
        for idx in range(34, 44):
            line = block[idx]
            
            vals = parse_e12(line, 6)
            vals_vert_drift.extend(vals)
            
        for i, val in enumerate(vals_vert_drift):
            try:
                timestamp = dt + timedelta(seconds = time_minute[i])
                vertical_drift.append([timestamp, val])
                
            except:
                continue
        
#%%create dataframes
df_ephemeris = pd.DataFrame(ephemeris,
                            columns = ['Time', 'Geographic lat (degrees north)', 'Geographic lon (degrees east)',
                                       'Apex lat (degrees north)', 'Apex lon (degrees east)', 'Apex local time (hours)', 'Satellite altitude (km)',])

#convert geographic values to aacgm and add the columns to the dataframe
aacgm_lats = []
aacgm_lons = []
aacgm_mlts = []

for i in range(len(ephemeris)):
    eTime = df_ephemeris['Time'][i] #time from the ephemeris dataframe
    aacgm.set_datetime(eTime.year, eTime.month, eTime.day, eTime.hour, eTime.minute, eTime.second) #sets the date for the IMF for the proper coordinates
    
    geo_lat = df_ephemeris['Geographic lat (degrees north)'][i] #geographic latitude
    geo_lon = df_ephemeris['Geographic lon (degrees east)'][i] #geographic longitude
    sat_alt = df_ephemeris['Satellite altitude (km)'][i] #altitude in km
    
    aacgm_lat, aacgm_lon, rad = aacgm.convert(geo_lat, geo_lon, sat_alt, 0) #returns AACGM latitude/longitude and geocentric radial distance in Re
    aacgm_lats.append(aacgm_lat)
    if aacgm_lon < 0:
        aacgm_lon = aacgm_lon + 360
        
    aacgm_lons.append(aacgm_lon)
    
    aacgm_mlt = aacgm.mlt_convert(eTime.year, eTime.month, eTime.day, eTime.hour, eTime.minute, eTime.second, aacgm_lon)
    aacgm_mlts.append(aacgm_mlt)
    
#add the aacgm columns to the ephemeris dataframe
df_ephemeris['AACGM lat'] = aacgm_lats
df_ephemeris['AACGM lon'] = aacgm_lons
df_ephemeris['AACGM MLT'] = aacgm_mlts

#interpolate ephemeris data
df_ephemeris = df_ephemeris.set_index('Time')
df_ephemeris_1s = df_ephemeris.resample('1s').asfreq()
df_ephemeris_1s = df_ephemeris_1s.interpolate(method = 'time') #ephemeris data at one second intervals

N = 86400 - len(df_ephemeris_1s)

st_time = df_ephemeris_1s.index[0]
nd_time = df_ephemeris_1s.index[-1] + pd.Timedelta(seconds = N)
time_index = pd.date_range(start = st_time, end = nd_time, freq = '1s')

df_ephemeris_1s = df_ephemeris_1s.reindex(time_index)
 
#==============================================================================
#create potential dataframe   
df_potential = pd.DataFrame(sat_potential,
                            columns = ['Time', 'Potential (volts)', f'Source ({source_potential_str})']).set_index('Time')

df_potential_1s = df_potential.resample('1s').asfreq()

N = 86400 - len(df_potential_1s)

st_time = df_potential_1s.index[0]
nd_time = df_potential_1s.index[-1] + pd.Timedelta(seconds = N)

df_potential_1s = df_potential_1s.reindex(time_index)

#==============================================================================
#create plasma density dataframe
df_density = pd.DataFrame(primary_density,
                          columns = ['Time', 'Primary plasma density (#/cm^3)', 'Source']).set_index('Time')

#==============================================================================
#create horizontal ion drift velocity dataframe
df_hori_drift = pd.DataFrame(horizontal_drift,
                             columns = ['Time', 'Horizontal ion drift (m/s)']).set_index('Time')

#==============================================================================
#create vertical ion drift velocity dataframe
df_vert_drift = pd.DataFrame(vertical_drift,
                             columns = ['Time', 'Vertical ion drift (m/s)']).set_index('Time')

#-0.10000E+38 -> missing float
#99999 -> missing integer

#%%create plot
st = pd.to_datetime(sttime, format = '%Y%m%d%H%M')
nd = pd.to_datetime(ndtime, format = '%Y%m%d%H%M')

fig = plt.figure(figsize = (14, 10))

gs = gridspec.GridSpec(4, 2,
                       width_ratios = [50, 0.5], #narrow column for colorbar, adjust to change thickness
                       wspace = 0.02, #adjust to change distance of colorbar from spectrogram
                       height_ratios = [1, 1, 1, 1],
                       hspace = 0.1)

ax_dens = fig.add_subplot(gs[0, 0])
ax_pot = fig.add_subplot(gs[1, 0], sharex = ax_dens)
ax_hdrift = fig.add_subplot(gs[2, 0], sharex = ax_dens)
ax_vdrift = fig.add_subplot(gs[3, 0], sharex = ax_dens)

x = time_index

ax_dens.plot(x, df_density['Primary plasma density (#/cm^3)'])
ax_dens.set_ylabel('Primary Plasma Density\n(number/cm$^3$)')
ax_dens.set_title(f'{sat} {date_str} {sttime[8:12]}-{ndtime[8:12]}', fontsize = '14')

ax_pot.scatter(x, df_potential_1s['Potential (volts)'], s = 1, color = 'black')
ax_pot.set_ylabel('Satellite Potential (volts)')

ax_hdrift.plot(x, df_hori_drift['Horizontal ion drift (m/s)'])
ax_hdrift.set_ylabel('Horizontal Ion\nDrift Velocities (m/s)')

ax_vdrift.plot(x, df_vert_drift['Vertical ion drift (m/s)'])
ax_vdrift.set_ylabel('Vertical Ion\nDrift Velocities (m/s)')

label_x = -0.03
labelsize = 10
label_pad = -9.5

#add and format labels on x-axis
ax_vdrift.xaxis.set_major_formatter(mdates.DateFormatter('%H%M')) #format the tick marks
ax_vdrift.set_xlim(st, nd)
xlabel_object = ax_vdrift.set_xlabel('UT', labelpad = label_pad)
xlabel_object.set_ha('right')
xlabel_object.set_position((label_x, 0))

axes = [ax_dens, ax_pot, ax_hdrift, ax_vdrift]
dfs = [df_density, df_potential_1s, df_hori_drift, df_vert_drift]
cols = ['Primary plasma density (#/cm^3)', 'Potential (volts)',
        'Horizontal ion drift (m/s)', 'Vertical ion drift (m/s)']

for ax in axes[:-1]:
    ax.tick_params(axis = 'x', length = 0) #remove tick marks from the underside of axes not on the bottom
    ax.tick_params(axis='x', labelbottom = False) #remove labels from underside of axes not on the bottom
    
for ax, df, col in zip(axes, dfs, cols):
    vals = df.loc[st:nd, col]
    ax.set_xlim(st, nd)

    ymin = vals.min()
    ymax = vals.max()
    pad  = 0.05 * (ymax - ymin)

    ax.set_ylim(ymin - pad, ymax + pad) #set y-limits based on the min/max values displayed in the plot
    
#==============================================================================
#add satellite geomagnetic latitude to x-axis
ax2 = ax_vdrift.secondary_xaxis('bottom')
ax2.set_xticks(ax_vdrift.get_xticks())
ax2.set_xticklabels(get_labels(ax_vdrift.get_xticks(), df_ephemeris_1s['AACGM lat'] , df_vert_drift))

#==============================================================================
#add MLT to x-axis
ax3 = ax_vdrift.secondary_xaxis('bottom')
ax3.set_xticks(ax_vdrift.get_xticks())
ax3.set_xticklabels(get_labels(ax_vdrift.get_xticks(), df_ephemeris_1s['AACGM MLT'] , df_vert_drift))

#==============================================================================
#add satellite altitude to x-axis
ax4 = ax_vdrift.secondary_xaxis('bottom')
ax4.set_xticks(ax_vdrift.get_xticks())
ax4.set_xticklabels(get_labels(ax_vdrift.get_xticks(), df_ephemeris_1s['Satellite altitude (km)'] , df_vert_drift))

x_axes = [ax2, ax3, ax4]
labels = ['MLAT', 'MLT', 'ALT']
spine_pos = 14
i = 0

for ax in x_axes:
    ax.spines['bottom'].set_position(('outward', spine_pos)) #set the spine position
    ax.spines['bottom'].set_visible(False) #hide the spine of the second axis
    ax.tick_params(axis = 'x', which = 'both', length = 0, labelsize = labelsize) #hide tick lines
    
    xlabel_object = ax.set_xlabel(labels[i], labelpad = label_pad)
    xlabel_object.set_ha('right')
    xlabel_object.set_position((label_x, 0))
    
    spine_pos += 12
    i += 1
    
save_path = '/import/SUPERDARN/matthew/dmsp/plots'
file_name = 'ssies_' + str(sat) + '_' + str(date_str) + '_' + str(sttime[8:12]) + '-' + str(ndtime[8:12])
plt.savefig(save_path + '/' + date_str + '/' + file_name + '.png')

    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    