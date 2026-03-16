#!/usr/bin/env python3

#import required modules
import re
import glob
import gzip
import aacgm
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from collections import Counter
from datetime import date, datetime, timedelta
#from pandas._libs.tslibs.timestamps import Timestamp

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

#define a function for decompressing the raw SSJ data
def decompress(I):
    X = I % 32
    Y = (I - X) // 32

    return (X + 32) * (2**Y) - 33

#%%set start date for reading specific ssies.gz file
sttime = str(20150217)
#ndtime = str(201111010700)
date_str = sttime[0:8] #year, month, and day string

sat = 'f18' #set satellites from f15 to f18
sat_num = sat[1:3]
if int(sat_num) > 15:
    sensor_type = 5 #sensor is ssj5
    del_t = 0.05 #accumulation time for ssj5

else:
    sensor_type = 4 #sensor is ssj4
    del_t = 0.098 #accumulation time for ssj4

ssies_data_path = '/import/SUPERDARN/matthew/dmsp/ssies_edr/' + sttime[0:8] + '/'
pattern = re.compile(rf"SIES\d-{sat}.*?\.GLOBAL_DD\.{date_str}_TP\.\d{{6}}-\d{{6}}_DF\.EDR\.gz$")

ssies_files = glob.glob(ssies_data_path + '*.gz') #finds all .gz files in the specified data path
for file in ssies_files:
    if sat_num in file: #find the file for the current satellite
        ssies_fname = file
        
#%%open ssies.gz file
blocks = []
block_size = 114

ephemeris = []
sat_potential = [] #the electric potential of the spacecraft relative to the surrounding plasma
primary_density = []
horizontal_drift = []
vertical_drift = []

with gzip.open(ssies_fname, 'rt') as f:
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
        #date = parts[3] #breaks when used due to 'date' function in datetime
        time = parts[4]
        
        dtm = datetime.strptime(date_str + time, '%Y%m%d%H%M') #convert the metadata to a datetime object
        
        #======================================================================
        #get and append ephemeris data for each block
        time_ephemeris = np.arange(0, 60, 20)
        time_ephemeris = time_ephemeris.tolist() #seconds each minute for ephemeris measurements
        
        for i, idx in enumerate([4,5,6]):
            try:
                timestamp = dtm + timedelta(seconds = time_ephemeris[i])
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
                timestamp = dtm + timedelta(seconds = time_potential[i])
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
                timestamp = dtm + timedelta(seconds = time_minute[i])
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
                timestamp = dtm + timedelta(seconds = time_minute[i])
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
                timestamp = dtm + timedelta(seconds = time_minute[i])
                vertical_drift.append([timestamp, val])
                
            except:
                continue
        
#%%create dataframes from ssies data
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

if len(df_density < 86400):
    df_density = df_density.resample('1s').asfreq()

#==============================================================================
#create horizontal ion drift velocity dataframe
df_hori_drift = pd.DataFrame(horizontal_drift,
                             columns = ['Time', 'Horizontal ion drift (m/s)']).set_index('Time')

if len(df_hori_drift < 86400):
    df_hori_drift = df_hori_drift.resample('1s').asfreq()

#==============================================================================
#create vertical ion drift velocity dataframe
df_vert_drift = pd.DataFrame(vertical_drift,
                             columns = ['Time', 'Vertical ion drift (m/s)']).set_index('Time')

if len(df_vert_drift < 86400):
    df_vert_drift = df_vert_drift.resample('1s').asfreq()

#%%open ssj.gz file
ssj_data_path = '/import/SUPERDARN/matthew/dmsp/ssj4/' + sttime[0:8] + '/'
fday = sttime[6:8] #day from given time
fmonth = sttime[4:6] #month from given time
fyear = sttime[0:4] #year from given time
year = fyear[2:4] #last two digits of year
day_of_year = date(int(fyear), int(fmonth), int(fday)).timetuple().tm_yday

if day_of_year < 100:
    day_of_year = f'0{day_of_year}'

ssj_fname = f'j{sensor_type}{sat}{year}{day_of_year}.gz'

with gzip.open(ssj_data_path + ssj_fname, 'rb') as f:
    raw_data = f.read()
    
data = np.frombuffer(raw_data, dtype = '>u2')
indices = np.where(data == 80)[0]
gaps = np.diff(indices)
common_gaps = Counter(gaps).most_common(10)

records = data.reshape(-1, 2640)

#get date values that are constant over the full 24 hours
date_obj = datetime.strptime(f'{fyear} {fday}', '%Y %j')

#set channel values and headers
raw_channel_order = [4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17] #order the energy channels appear in the .gz file
channel_energies = np.array([30000, 24000, 13900, 9450, 6460, 4400, 3000, 2040, 1392, 949, 949, 646, 440, 300, 204, 139, 95, 65, 44, 30]) #channel energies in eV
deltaE_i = np.array([9600, 8050, 5475, 3720, 2525, 1730, 1180, 804, 545.5, 373, 373, 254.5, 173, 118, 80.5, 54.5, 37, 25.5, 17.5, 14])
energy_headers = ['seconds'] + [f'channel {i} ({channel_energies[i-1]} eV)' for i in raw_channel_order] #electron energy channel headers

#%%make geometric factors dataframe
sat_headers = ['channel', 'f16', 'f17', 'f18', 'f19', 'f20']

factor_values = np.array([[1.781, 1.044, 0.725, 3.735, 2.992],
                         [1.477, 0.808, 0.534, 2.885, 2.101],
                         [1.188, 0.602, 0.412, 2.196, 1.532],
                         [0.935, 0.458, 0.315, 1.615, 1.080],
                         [0.722, 0.349, 0.266, 1.170, 0.782],
                         [0.551, 0.262, 0.199, 0.832, 0.539],
                         [0.416, 0.191, 0.147, 0.605, 0.389],
                         [0.306, 0.142, 0.107, 0.418, 0.295],
                         [0.225, 0.103, 0.0803, 0.280, 0.186],
                         [0.166, 0.0727, 0.0526, 0.197, 0.128],
                         [None, None, None, None, None],
                         [0.123, 0.0541, 0.041, 0.134, 0.0825],
                         [0.0876, 0.0394, 0.0296, 0.0958, 0.0516],
                         [0.0613, 0.0394, 0.0296, 0.0640, 0.0351],
                         [0.0429, 0.0188, 0.014, 0.0445, 0.0235],
                         [0.0289, 0.0134, 0.0104, 0.0312, 0.0175],
                         [0.0182, 0.00901, 0.00708, 0.0204, 0.00975],
                         [0.0113, 0.00645, 0.00562, 0.00830, 0.00723],
                         [0.00621, 0.00445, 0.00386, 0.00222, 0.00410],
                         [0.00307, 0.00294, 0.00239, 0.000639, 0.00193]])
    
channel = np.arange(1, 21, 1)

factor_values = np.column_stack((channel, factor_values))
geometric_factors = pd.DataFrame(factor_values, columns = sat_headers)
geometric_factors.set_index('channel', inplace = True)
geometric_factors.index.name = 'channel'

sat_factors = geometric_factors[f'{sat}']

#%%loop through each minute
full_energy_flux = []
full_number_flux = []

for i in range(len(records)):
    start_idx = i * 60
    end_idx = start_idx + 60
    single_minute = records[i, :] #data from a specified minute
    second_data = single_minute[17:]
    second_data = second_data.reshape(-1, 43) #electron and ion channel energies
    electron_data = second_data[:, :21] #electron channel energies
    
    #%%make electron counts dataframe
    hour = single_minute[1]
    minute = single_minute[2]

    date_obj = dt.datetime(int(fyear), int(fmonth), int(fday), hour, minute)

    seconds = second_data[:, 0] // 1000 #convert time from milliseconds to seconds
    date_obj = np.datetime64(date_obj)
    time_array = (date_obj + np.timedelta64(1, 's') * seconds).astype(object)
    time_array = time_array[:-1]

    electron_raw = pd.DataFrame(electron_data, columns = energy_headers)
    electron_raw['seconds'] = seconds
    electron_raw.set_index('seconds', inplace = True)
    electron_raw.index.name = 'seconds'
    electron_raw = electron_raw.iloc[:-1]

    #reorder the energy channels
    desired_headers = [f'channel {i} ({channel_energies[i-1]} eV)' for i in range(1, 21)]
    electron_raw = electron_raw[desired_headers]
    electron_raw = electron_raw.rename(columns = {'channel 11 (949 eV)': 'Status Word 1'})
    electron_counts = decompress(electron_raw)
    
    #%%calculate the differential number/energy fluxes
    denominator = sat_factors * del_t
    diff_number_flux = electron_counts / denominator.values #calculate the differential number flux
    
    diff_number_flux[electron_counts <= 2] = 0 #set to filter instrument noise

    diff_energy_flux = diff_number_flux * channel_energies #calculate the differential energy flux
    diff_energy_flux.index = date_obj + pd.to_timedelta(diff_energy_flux.index, unit = 's') #add the date object to the seconds
    diff_energy_flux.index.name = 'time' #rename the dataframe index to time
    
    full_energy_flux.append(diff_energy_flux)
    full_number_flux.append(diff_number_flux)
    
df_diff_energy_flux = pd.concat(full_energy_flux, axis = 0)
df_diff_number_flux = pd.concat(full_number_flux, axis = 0)

J_i = df_diff_number_flux #differential number flux
JE_i = df_diff_energy_flux #differential energy flux
    
J = np.nansum(J_i * deltaE_i, axis = 1) #integrated number flux
JE = np.nansum(JE_i * deltaE_i, axis = 1) #integrated energy flux
E_avg = np.divide(
    JE,
    J,
    out = np.full_like(J, np.nan),
    where = J > 0)

df_E_avg =  pd.DataFrame(E_avg, index = df_diff_energy_flux.index)
df_E_avg = df_E_avg.rename(columns = {0: 'Average electron energy (eV)'})

#remove the 11th column of necessary arrays if SSJ5 data is being used
if int(sat_num) > 15:
    channel_energies = np.delete(channel_energies, 10)
    df_diff_energy_flux = df_diff_energy_flux.drop(columns = ['Status Word 1'])

#fill in missing times if necessary    
if len(df_diff_energy_flux < 86400):
    df_diff_energy_flux = df_diff_energy_flux[~df_diff_energy_flux.index.duplicated(keep = 'first')] #remove duplicate indicies
    df_diff_energy_flux = df_diff_energy_flux.resample('1s').asfreq()
    
if len(df_E_avg < 86400):
    df_E_avg = df_E_avg[~df_E_avg.index.duplicated(keep = 'first')] #remove duplicate indicies
    df_E_avg = df_E_avg.resample('1s').asfreq()

#%%define geomagnetic/geographic spatial window
max_mlat = 75 #maximum geomagnetic latitude in aacgm
min_mlat = 60 #minimum geomagnetic latitude in aacgm

max_glon = 310 #maximum geographic latitude (degrees east)
min_glon = 190 #minimum geographic latitude (degrees east)

max_mlt = 23.99 #maximum MLT
min_mlt = 19.00 #minimum MLT

#mask= ((df_ephemeris_1s['AACGM lat'] >= min_mlat) &
#       (df_ephemeris_1s['AACGM lat'] <= max_mlat))

mask_mlat = df_ephemeris_1s['AACGM lat'].between(min_mlat, max_mlat)
mask_glon = df_ephemeris_1s['Geographic lon (degrees east)'].between(min_glon, max_glon)

mask = mask_mlat & mask_glon

subset_ephemeris = df_ephemeris_1s[mask] #create a dataframe for where the satellite falls within the defined geomagnetic latitude range
time_diff = subset_ephemeris.index.to_series().diff() #get the time difference between data points
gap_threshold = pd.Timedelta(seconds = 2) #set a threshold for the gap between data blocks

breaks = time_diff > gap_threshold
ids = breaks.cumsum()

for _, group in subset_ephemeris.groupby(ids):
    block_start = group.index[0]
    block_end = group.index[-1]
    
    st_hhmm = block_start.strftime("%H%M")
    nd_hhmm = block_end.strftime("%H%M")
    
    print(f'\nblock start: {block_start}')
    print(f'block end: {block_end}')

#%%create plot
#st = pd.to_datetime(sttime, format = '%Y%m%d%H%M')
#nd = pd.to_datetime(ndtime, format = '%Y%m%d%H%M')

    fig = plt.figure(figsize = (20, 14))

    gs = gridspec.GridSpec(5, 2,
                           width_ratios = [50, 0.6], #narrow column for colorbar, adjust to change thickness
                           wspace = 0.02, #adjust to change distance of colorbar from spectrogram
                           height_ratios = [1, 1, 1, 1, 1],
                           hspace = 0.1)

    ax_spec = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1]) #colorbar axis for spectrogram

    ax_mean = fig.add_subplot(gs[1, 0], sharex = ax_spec)
    ax_dens = fig.add_subplot(gs[2, 0], sharex = ax_spec)
    ax_hdrift = fig.add_subplot(gs[3, 0], sharex = ax_spec)
    ax_vdrift = fig.add_subplot(gs[4, 0], sharex = ax_spec)

    x = time_index

    #create the spectrogram
    diff_energy_flux_arr = df_diff_energy_flux.values.astype(float)
    diff_energy_flux_log10 = np.log10(diff_energy_flux_arr)
    im = ax_spec.pcolormesh(
        x, channel_energies, diff_energy_flux_log10.T, #x, y, and z-axes
        cmap = 'jet', shading = 'nearest',
        vmax = np.max(10)) #define color limit

    ax_spec.set_yscale('log')
    ax_spec.set_ylabel('Electron Channels\nlog (eV)')
    ax_spec.set_title(f'{sat} {date_str} {st_hhmm}-{nd_hhmm}', fontsize = '14') #set plot title

    cbar = fig.colorbar(im, cax = cax)
    cbar.set_label('Differential Energy Flux\nlog (eV/$cm^2$-s-sr-eV)')
    
    #plot average electron energies
    ax_mean.scatter(x, df_E_avg, s = 10, color = 'black')
    ax_mean.set_yscale('log')
    ax_mean.set_ylabel('Average Electron\nEnergy (eV)')

    #plot plasma density
    ax_dens.plot(x, df_density['Primary plasma density (#/cm^3)'], color = 'red', linewidth = 3)
    ax_dens.set_ylabel('Primary Plasma Density\n(number/cm$^3$)')


    #plot horizontal ion drifts
    ax_hdrift.plot(x, df_hori_drift['Horizontal ion drift (m/s)'], linewidth = 3)
    ax_hdrift.set_ylabel('Horizontal Ion\nDrift Velocities (m/s)')
    ax_hdrift.axhline(0, color = 'black', linestyle = '--', linewidth = 1)

    #plot vertical ion drifts
    ax_vdrift.plot(x, df_vert_drift['Vertical ion drift (m/s)'], linewidth = 3)
    ax_vdrift.set_ylabel('Vertical Ion\nDrift Velocities (m/s)')
    ax_vdrift.set_xlim(block_start, block_end) #set x limits based on the current block
    ax_vdrift.axhline(0, color = 'black', linestyle='--', linewidth = 1)

    label_x = -0.02
    labelsize = 10
    label_pad = -9.5

    #add and format labels on bottom x-axis
    ax_vdrift.xaxis.set_major_formatter(mdates.DateFormatter('%H%M')) #format the tick marks
    xlabel_object = ax_vdrift.set_xlabel('UT', labelpad = label_pad)
    xlabel_object.set_ha('right')
    xlabel_object.set_position((label_x, 0))

    axes = [ax_spec, ax_mean, ax_dens, ax_hdrift, ax_vdrift]
    dfs = [df_E_avg, df_density, df_hori_drift, df_vert_drift]
    cols = ['Average electron energy (eV)', 'Primary plasma density (#/cm^3)',
        'Horizontal ion drift (m/s)', 'Vertical ion drift (m/s)']

    for ax in axes[:-1]:
        ax.tick_params(axis = 'x', length = 0) #remove tick marks from the underside of axes not on the bottom
        ax.tick_params(axis='x', labelbottom = False) #remove labels from underside of axes not on the bottom
    
    for ax, df, col in zip(axes[1:], dfs, cols):
        vals = df.loc[block_start:block_end, col]
        
        ymin = vals.min()
        ymax = vals.max()
        pad  = 0.05 * (ymax - ymin)
        
        if np.isnan(ymax):
            ymax = df[col].max()
            
        if np.isnan(ymin):
            ymin = df[col].min()
            
        if np.isnan(pad):
            pad = 0
        
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
    spine_pos = 16
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
        
    block_times = pd.date_range(start = block_start, end = block_end,
                                periods = 10)
        
    #add vertical lines to subplots
    for t in block_times:
        for ax in axes:    
            ax.axvline(t, color = 'black',
                       linestyle = '--', linewidth=1.2,
                       alpha = 0.6, zorder = 1)
    
    save_path = '/import/SUPERDARN/matthew/dmsp/plots'
    file_name = sat + '_' + date_str + '_' + st_hhmm + '-' + nd_hhmm
    plt.savefig(save_path + '/' + date_str + '/' + sat + '/' + file_name + '.png')    
    plt.show()    
    plt.close()
    












