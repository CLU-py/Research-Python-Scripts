#!/usr/bin/env python3

#import required modules
import gzip
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from datetime import date
from collections import Counter

#%%define necessary functions
def get_labels(tick_locs, data_array, dataframe):
    tick_datetimes = mdates.num2date(tick_locs)
    labels = []
    for t in tick_datetimes:
        target = t.replace(tzinfo = None)
        idx = dataframe.index.get_indexer([target], method = 'nearest')[0]
        val = data_array[idx]
        labels.append(f"{val:.1f}" if not np.isnan(val) else "")
    return labels

#define a function for decompressing the raw data
def decompress(I):
    X = I % 32
    Y = (I - X) // 32

    return (X + 32) * (2**Y) - 33

#%%set start date for reading specific .gz file
#time = str(20140327)
sttime = str(201403270500)
ndtime = str(201403270700)
date_str = sttime[0:8] #year, month, and day string

st_hhmm = sttime[8:12]
nd_hhmm = ndtime[8:12]

data_path = '/import/SUPERDARN/matthew/dmsp/ssj4/' + sttime[0:8]
fday = sttime[6:8] #day from given time
fmonth = sttime[5:6] #month from given time
fyear = sttime[0:4] #year from given time
year = fyear[2:4] #last two digits of year
day_of_year = date(int(fyear), int(fmonth), int(fday)).timetuple().tm_yday

if day_of_year < 100:
    day_of_year = f'0{day_of_year}'

sat = 'f17'
sat_num = sat[1:3]
if int(sat_num) > 15:
    sensor_type = 5 #sensor is ssj5
    del_t = 0.05 #accumulation time for ssj5

else:
    sensor_type = 4 #sensor is ssj4
    del_t = 0.098 #accumulation time for ssj4

fname = f'j{sensor_type}{sat}{year}{day_of_year}.gz'

with gzip.open(fname, 'rb') as f:
    raw_data = f.read()
    
data = np.frombuffer(raw_data, dtype = '>u2')
indices = np.where(data == 80)[0]
gaps = np.diff(indices)
common_gaps = Counter(gaps).most_common(10)

records = data.reshape(-1, 2640)

#%%plot columns and variance to confrim shape of records array
#plt.figure(figsize = (10, 12))
#plot the first 500 records and all 264 columns
#plt.imshow(records, aspect = 'auto', cmap = 'gist_stern', interpolation = 'nearest')
#plt.title('Data Layout (Width 264)')
#plt.xlabel('Word Index (0-2639)')
#plt.ylabel('Record Index')
#plt.colorbar(label = 'Value')

#calculate variance for every one of the 2640 columns
#col_variance = np.var(records.astype(float), axis = 0)

#plt.figure(figsize = (15, 5))
#plt.plot(col_variance)
#plt.title('Variance Profile of the 2640-word Row')
#plt.xlabel('Word Index (0-2639)')
#plt.ylabel('Variance (Activity)')
#plt.grid(True)
#plt.show()

#%%create empty arrays for values over the entire 24 hour span
total_seconds = len(records) * 60
full_altitude = np.full(total_seconds, np.nan) #empty array for satellite altitude (km)
full_mlt = np.full(total_seconds, np.nan) #empty array for satellite MLT
full_mlat = np.full(total_seconds, np.nan) #empty array for satellite geomagnetic latitude
full_utc = np.full(total_seconds, np.nan, dtype = 'U4') #empty array for UTC
full_energy_flux = []
full_number_flux = []

#get date values that are constant over the full 24 hours
date_obj = datetime.datetime.strptime(f'{fyear} {fday}', '%Y %j')

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
for i in range(len(records)):
    start_idx = i * 60
    end_idx = start_idx + 60
    single_minute = records[i, :] #data from a specified minute
    second_data = single_minute[17:]
    second_data = second_data.reshape(-1, 43) #electron and ion channel energies
    electron_data = second_data[:, :21] #electron channel energies
    position = single_minute[5:13]

    #%%get satellite position
    if position[5] > 1800:
        aacgm_lat = (position[5] - 4995) / 10 #satellite geomagnetic latitude
    else:    
        aacgm_lat = (position[5] - 900) / 10 #satellite geomagnetic latitude
    
    aacgm_ltime = position[7] #satellite MLT
    altitude = position[2] * 1.852 #satellite altitude in km

    aacgm_lat = np.full(60, aacgm_lat) #set the geomagnetic latitude value for the whole minute
    aacgm_ltime = np.full(60, aacgm_ltime) #set the MLT for the whole minute
    altitude = np.full(60, altitude) #set the altitude for the whole minute
    
    full_altitude[start_idx:end_idx] = altitude
    full_mlt[start_idx:end_idx] = aacgm_ltime
    full_mlat[start_idx:end_idx] = aacgm_lat

    #%%make electron counts dataframe
    hour = single_minute[1]
    minute = single_minute[2]

    date_obj = datetime.datetime(int(fyear), int(fmonth), int(fday), hour, minute)

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
    
diff_energy_flux_df = pd.concat(full_energy_flux, axis = 0)
diff_number_flux_df = pd.concat(full_number_flux, axis = 0)

J_i = diff_number_flux_df #differential number flux
JE_i = diff_energy_flux_df #differential energy flux
    
J = np.nansum(J_i * deltaE_i, axis = 1) #integrated number flux
JE = np.nansum(JE_i * deltaE_i, axis = 1) #integrated energy flux
E_avg = np.divide(
    JE,
    J,
    out = np.full_like(J, np.nan),
    where = J > 0)

E_avg =  pd.DataFrame(E_avg, index = diff_energy_flux_df.index)

#remove the 11th column of necessary arrays if SSJ5 data is being used
if int(sat_num) > 15:
    channel_energies = np.delete(channel_energies, 10)
    diff_energy_flux_df = diff_energy_flux_df.drop(columns=['Status Word 1'])
    
#%%convert start and end strings to datetime
dt_start = pd.to_datetime(sttime, format = '%Y%m%d%H%M')
dt_end   = pd.to_datetime(ndtime, format = '%Y%m%d%H%M')
duration_minutes = (dt_end - dt_start).total_seconds() / 60 #calculate the total duration in minutes

plot_index = pd.date_range(start = sttime, end = ndtime, freq = 's') #time indicies based on the start time and end time
idx = np.where((diff_energy_flux_df.index >= dt_start) & (diff_energy_flux_df.index <= dt_end))[0] #indices of set time interval

#replace missing data points with correct times and nan values
df_interval = diff_energy_flux_df.loc[dt_start:dt_end] #set the interval to plot based on the start and end times
mask = df_interval.any(axis = 1)
false_idx = np.where(~mask)[0]
df_interval = df_interval[df_interval.any(axis = 1)] #remove any values that are all zeros; these indicate data gaps
df_plot = df_interval.reindex(plot_index) #fill in the missing data timestamps with nans

E_avg_interval = E_avg.loc[dt_start:dt_end]

diff_energy_flux_array = df_plot.values.astype(float) #get the differential energy flux values from the interval dataframe

#size position arrays using interval dataframe
try:
    num_nans = len(df_plot) - len(mask) #number of nans to be added to the position arrays
    ins_idx = false_idx[0] #index to insert the specified number of nans to
    nan_block = np.full(num_nans, np.nan)

    plot_mlt = full_mlt[idx] #values of MLT to be plotted
    plot_mlt[false_idx] = np.nan
    plot_mlt = np.insert(plot_mlt, ins_idx, nan_block)

    plot_mlat = full_mlat[idx] #values of geomagnetic latitude to be plotted
    plot_mlat[false_idx] = np.nan
    plot_mlat = np.insert(plot_mlat, ins_idx, nan_block)

    plot_altitude = full_altitude[idx] #values of altitude to be plotted
    plot_altitude[false_idx] = np.nan
    plot_altitude = np.insert(plot_altitude, ins_idx, nan_block)

except:
    plot_mlt = full_mlt[idx]
    plot_mlat = full_mlat[idx]
    plot_altitude = full_altitude[idx]
    #print('no missing timestamps')

#%%create the time energy sepectrogram
#fig, (ax_spec, ax_mean) = plt.subplots(2, 1,
#                                       figsize=(20, 8),
#                                       sharex=True) #specify axes for the spectrogram and the average electron energies

fig = plt.figure(figsize=(20, 8))

gs = gridspec.GridSpec(
    2, 2,
    width_ratios = [50, 0.5], #narrow column for colorbar, adjust to change thickness
    wspace = 0.02, #adjust to change distance of colorbar from spectrogram
    height_ratios = [1, 1],
    hspace = 0.05
)

ax_spec = fig.add_subplot(gs[0, 0])
ax_mean = fig.add_subplot(gs[1, 0], sharex = ax_spec)
cax = fig.add_subplot(gs[0, 1])  #colorbar axis

x = df_plot.index #number of points to plot on x-axis
diff_energy_flux_log10 = np.log10(diff_energy_flux_array)

im = ax_spec.pcolormesh(
    x, channel_energies, diff_energy_flux_log10.T, #x, y, and z-axes
    cmap = 'jet', shading = 'nearest',
    vmax = np.max(10)) #define color limit

label_x = -0.01
labelsize = 10
pad = -9.5

ax_mean.plot(df_plot.index, E_avg_interval,
             color = 'black', linewidth = 1.5)

#add labels to UT axis
#==============================================================================
ax_mean.tick_params(axis = 'x', direction = 'in', length = 6, labelsize = labelsize) #place tick marks on inside of spine and size accordingly
ax_mean.xaxis.set_major_formatter(mdates.DateFormatter('%H%M')) #format the tick marks
ax_mean.xaxis.set_major_locator(mdates.MinuteLocator(interval = 9)) #set the tick interval in minutes
#locator = mdates.AutoDateLocator(minticks = 5, maxticks = 22)
#ax.xaxis.set_major_locator(locator)

xlabel_object = ax_mean.set_xlabel('UT', labelpad = pad)
xlabel_object.set_ha('right')
xlabel_object.set_position((label_x, 0))

#add satellite geomagnetic latitude to x-axis
#==============================================================================
ax2 = ax_mean.secondary_xaxis('bottom')
ax2.set_xticks(ax_mean.get_xticks())
#ax2.set_xticklabels([f'{m:.1f}' for m in plot_mlat[ax.get_xticks().astype(int)]])
ax2.set_xticklabels(get_labels(ax_mean.get_xticks(), plot_mlat, df_plot))

ax2.spines['bottom'].set_position(('outward', 12)) #set the spine position
ax2.spines['bottom'].set_visible(False) #hide the spine of the second axis
ax2.tick_params(axis = 'x', which = 'both', length = 0, labelsize = labelsize) #hide tick lines

xlabel_object = ax2.set_xlabel('MLAT', labelpad = pad)
xlabel_object.set_ha('right')
xlabel_object.set_position((label_x, 0))

#add MLT to x-axis
#==============================================================================
ax3 = ax_mean.secondary_xaxis('bottom')
ax3.set_xticks(ax_mean.get_xticks())
#ax3.set_xticklabels([f'{m:.1f}' for m in plot_mlt[ax.get_xticks().astype(int)]])
ax3.set_xticklabels(get_labels(ax_mean.get_xticks(), plot_mlt, df_plot))

ax3.spines['bottom'].set_position(('outward', 24)) #set the spine position
ax3.spines['bottom'].set_visible(False) #hide the spine of the second axis
ax3.tick_params(axis = 'x', which = 'both', length = 0, labelsize = labelsize) #hide tick lines

xlabel_object = ax3.set_xlabel('MLT', labelpad = pad)
xlabel_object.set_ha('right')
xlabel_object.set_position((label_x, 0))

#add satellite altitude to x-axis
#==============================================================================
ax4 = ax_mean.secondary_xaxis('bottom')
ax4.set_xticks(ax_mean.get_xticks())
#ax4.set_xticklabels([f'{m:.1f}' for m in plot_altitude[ax.get_xticks().astype(int)]])
ax4.set_xticklabels(get_labels(ax_mean.get_xticks(), plot_altitude, df_plot))

ax4.spines['bottom'].set_position(('outward', 36)) #set the spine position
ax4.spines['bottom'].set_visible(False) #hide the spine of the second axis
ax4.tick_params(axis = 'x', which = 'both', length = 0, labelsize = labelsize) #hide tick lines

xlabel_object = ax4.set_xlabel('Alt', labelpad = pad)
xlabel_object.set_ha('right')
xlabel_object.set_position((label_x, 0))

#set plot title
ax_spec.set_title(f'{sat} {fyear}{fmonth}{fday} {st_hhmm}-{nd_hhmm}')

ax_spec.set_ylabel('Electron Energy\nlog (eV)')
ax_spec.set_yscale('log')
ax_spec.tick_params(axis = 'x', length = 0) #remove tick marks from the underside of the spectrogram x-axis
ax_spec.tick_params(axis='x', labelbottom = False)
ax_spec.tick_params(axis = 'y', which = 'major', direction = 'in', left = True, right = True, length = 6)
ax_spec.tick_params(axis = 'y', which = 'minor', direction = 'in', left = True, right = True, length = 3)

ax_mean.set_ylabel('Eavg (eV)')
ax_mean.set_yscale('log')
ax_mean.tick_params(axis = 'y', which = 'major', direction = 'in', left = True, right = True, length = 6)
ax_mean.tick_params(axis = 'y', which = 'minor', direction = 'in', left = True, right = True, length = 3)

#cbar = fig.colorbar(im, ax = ax_spec, orientation = 'vertical')
cbar = fig.colorbar(im, cax = cax)
cbar.set_label('Differential Energy Flux\nlog (eV/$cm^2$-s-sr-eV)')

#save_path = '/import/SUPERDARN/matthew/dmsp/plots'
#file_name = 'gz_' + str(sat) + '_' + str(date_str) + '_' + str(sttime[8:12]) + '-' + str(ndtime[8:12])
#plt.savefig(save_path + '/' + date_str + '/' + file_name + '.png')

#plt.plot(E_avg)
#plt.yscale('log')















