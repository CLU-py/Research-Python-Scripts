#!/usr/bin/env python3
#TO MAKE GIF FROM IMAGES GENERATED:
#IN DIRECTORY WITH IMAGES RUN: convert -delay 10 -loop 0 *.png {file_name}.gif
#OR: ffmpeg -framerate 3 -pattern_type glob -i "*.png" -loop 0 {file_name}.gif

#TIPS
#TO COMMENT/UNCOMMENT SECTIONS OF CODE: Ctrl + /

#import required modules
import aacgm
import socket
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import cartopy .crs as ccrs
import matplotlib.path as mpath
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from map_grid import make_grid

#%%add commands to call in terminal
parser = argparseparser = argparse.ArgumentParser(description='sector_gdf_vel argument parser')
parser.add_argument('--line_plot', action = 'store_true') #command to generate line plots
parser.add_argument('--vectors', action = 'store_true') #command to generate velocity plots using vectors instead of magnitude

args = parser.parse_args()
line_plot = args.line_plot
vectors = args.vectors

#%%read text files into script and create dictionaries for each data set
#========================================================================================
#========================================================================================
hostname = socket.gethostname()
# event_dates = ['20110302', '20111101',
#               '20130202', '20130316',
#               '20140303', '20140321',
#               '20140326', '20140327',
#               '20150217', '20151004']

event_dates = ['20110302', '20111101',
              '20120219', '20130202',
              '20130316', '20140321',
              '20140326', '20140327',
              '20150217', '20151004']

#event_dates = ['20140303']

if hostname == 'wilcox.met.psu.edu':
    file_path = '/import/SUPERDARN/matthew'
    
else:
    file_path = 'filepath on PC'
    
df_dictionary = {}
location_dictionary = {}

print(f'\n{'='*120}\nreading text files\n{'='*120}')
for event in event_dates:
    data = []
    
    try:
        with open(f'{file_path}/{event}/data_text_new/{event}_text.txt') as file:
            headers = file.readline().strip().split(', ')
            keyname = event
            
            print(f'reading data from {event}')
            file.readline()
            substorm_location = file.readline().strip()
            
            for line in file:
                line = line.strip()
                if line == '':
                    continue
                
                if line[0].isdigit() and ':' in line:
                    current_time = line
            
                parts = line.split()
                if len(parts) == 9:
                    center_lat, event_centerlon = parts[0], parts[1]
                    velocity = parts[4]
                    velx = parts[5]
                    vely = parts[6]
                    brightness = parts[7]
                    covariance = parts[8]
                
                    data.append([current_time, center_lat, event_centerlon, velocity, velx, vely, brightness, covariance])
                
                df = pd.DataFrame(data, columns = ['time', 'center latitude', 'center longitude', 'velocity', 'velx', 'vely', 'brightness', 'covariance'])
                df_dictionary[keyname] = df
                
                location_dictionary[keyname] = substorm_location

    except Exception as e:
        print(f'the following error occured: {e}')
        
#%%define region for superposition grid
#========================================================================================
#========================================================================================
#find average latitude for all evetns
event_lats = [float(value.split()[0]) for value in location_dictionary.values()] #latitudes of each events substorm onset
avg_lat = sum(event_lats)/len(event_lats) #average latitude across all events
print(f'\naverage substorm onset latitude: {avg_lat}')

event_lons = [float(value.split()[1]) for value in location_dictionary.values()]

#find average date for all events, used for calculating region for new grid
date_objs = [datetime.strptime(event, '%Y%m%d') for event in event_dates]
timestamps = [date.timestamp() for date in date_objs]
avg_timestamp = sum(timestamps)/len(timestamps)
date_avg = datetime.fromtimestamp(avg_timestamp) #date that falls in the middle of all substorm events

print('average date of all substorms:', date_avg)

#set boundaries for new grid based on latitude and MLT ranges
min_lat = avg_lat - 5
max_lat = avg_lat + 6

northlat = max_lat
southlat = min_lat

#get longitude 2MLT away from 0 longitude at specific latitude
center_MLT = aacgm.mlt_convert(date_avg.year, date_avg.month, date_avg.day, 0, 0, 0, 0)
east_MLT = center_MLT + 2
west_MLT = center_MLT - 5

eastlon = aacgm.inv_mlt_convert(date_avg.year, date_avg.month, date_avg.day, 0, 0, 0, east_MLT)
westlon = aacgm.inv_mlt_convert(date_avg.year, date_avg.month, date_avg.day, 0, 0, 0, west_MLT) - 5

master_centerlon = round((eastlon + westlon)/2, 2) #center longitude of superposition grid

if eastlon < 0:
    eastlon = eastlon + 360
        
if westlon < 0:
    westlon = westlon + 360

if master_centerlon < 0:
    master_centerlon = master_centerlon + 360

diff = 360 - master_centerlon

eastlon = eastlon - diff
westlon = westlon - diff

#%%create superposition grid
#========================================================================================
#========================================================================================
dtor = np.pi/180
dlat = 1

nlats = int((northlat - southlat)/dlat) #number of latitude points
#dlon = dlat/np.cos(lat_min * dtor) #change in longitude

lats = []
lons = []

index = 0
lat_index = 0
lon_index = 0
cells = {'index':[], 'lat_index':[], 'lon_index':[], 'lat_ll':[], 'lon_ll':[],
        'lat_lr':[], 'lon_lr':[], 'lat_ul':[], 'lon_ul':[], 'lat_ur':[], 'lon_ur':[]}

for jt in range(nlats):
    lat_l = southlat + jt * dlat
    lat_u = southlat + (jt+1) * dlat
    lat = (lat_l + lat_u)/2

    dlon0 = dlat/np.cos(lat * dtor) #initial change in longitude at lowest latitude
    nlons = int(np.round(360/dlon0))
    dlon = 360/nlons
    lon0 = (dlon - dlon0)/2
    #lon_index = 0

    for jn in range(nlons):
        lon = lon0 + jn * dlon
        cells["index"].append(index)
        cells["lat_index"].append(lat_index)
        cells["lon_index"].append(lon_index)

        cells["lat_ll"].append(lat_l) #latitude in lower left of grid
        cells["lon_ll"].append(lon) #longitude in lower left of grid
            
        cells["lat_lr"].append(lat_l) #latitude in lower right of grid
        cells["lon_lr"].append(lon + dlon) #longitude in lower right of grid
            
        cells["lat_ul"].append(lat_u) #latitude in upper left of grid
        cells["lon_ul"].append(lon) #longitude in upper left of grid
            
        cells["lat_ur"].append(lat_u) #latitude in upper right of grid
        cells["lon_ur"].append(lon + dlon) #longitude in upper right of grid

        index += 1 #step foward in indicies
        lon_index += 1
        lat_index += 1

#make the cells numpy arrays
cells["index"] = np.asarray(cells["index"])
cells["lat_index"] = np.asarray(cells["lat_index"])
cells["lon_index"] = np.asarray(cells["lon_index"])
    
cells["lat_ll"] = np.asarray(cells["lat_ll"])
cells["lat_lr"] = np.asarray(cells["lat_lr"])

cells["lat_ul"] = np.asarray(cells["lat_ul"])
cells["lat_ur"] = np.asarray(cells["lat_ur"])

cells["lon_ll"] = np.asarray(cells["lon_ll"])
cells["lon_lr"] = np.asarray(cells["lon_lr"])
    
cells["lon_ul"] = np.asarray(cells["lon_ul"])
cells["lon_ur"] = np.asarray(cells["lon_ur"])

#========================================================================================
#========================================================================================
lat_ll = cells['lat_ll'] #list of lower left latitudes
lat_ul = cells['lat_ul'] #list of upper left latitudes

lon_ll = cells['lon_ll'] #list of lower left longitudes
lon_lr = cells['lon_lr'] #list of lower right longitudes

grid_points = len(cells['index'])

#set region based on superposition grid
if eastlon < westlon:
    east_region = lon_lr <= eastlon
    west_region = lon_ll >= westlon

    lat_region = (lat_ll >= southlat) & (lat_ll <= northlat)
    lon_region = np.logical_or(east_region, west_region)
    region_indx = np.logical_and(lat_region, lon_region)

else:
    region_indx = (lat_ll >= southlat) & (lat_ll <= northlat) & (lon_ll >= westlon) & (lon_lr <= eastlon)
    
#%%create arrays for master grid
master_lat_ll = []
master_lat_ul = []
master_lon_ll = []
master_lon_lr = []

for lat1, lat2, lon1, lon2 in zip(lat_ll[region_indx], lat_ul[region_indx], lon_ll[region_indx], lon_lr[region_indx]):
    master_lat_ll.append(lat1) 
    master_lat_ul.append(lat2)
    master_lon_ll.append(lon1)
    master_lon_lr.append(lon2)
    
master_lat_ll = np.array(master_lat_ll)
master_lat_ul = np.array(master_lat_ul)
master_lon_ll = np.array(master_lon_ll)
master_lon_lr = np.array(master_lon_lr)
    
master_index = len(master_lat_ll)

#%%superposition events

#create empty arrays for line plots at specific locations
if line_plot:
    onset_bright = [] #brightness value at substorm onset
    onset_vel = [] #brightness value at substorm onset
    onset_velx = []
    onset_vely = []
    onset_cov = [] #covariance value at substorm onset

    pole_bright = [] #brightness value poleward of substorm onset
    pole_vel = [] #velocity value poleward of substorm onset
    pole_velx = []
    pole_vely = []
    pole_cov = [] #covariance value poleward of substorm onset

    eq_bright = [] #brightness value equatorward of substorm onset
    eq_vel = [] #velocity value equatorward of substorm onset
    eq_velx = []
    eq_vely = []
    eq_cov = [] #covariance value equatorward of substorm onset

    west_bright = [] #brightness value west of substorm onset
    west_vel = [] #velocity value west of substorm onset
    west_velx = []
    west_vely = []
    west_cov = [] #covariance value west of substorm onset

    west_eq_bright = [] #brightness value west and equatorward of substorm onset
    west_eq_vel = [] #velocity value west and equatorward of substorm onset
    west_eq_velx = []
    west_eq_vely = []
    west_eq_cov = [] #covariance value west and equatorward of substorm onset

number = 0 #index for event latitudes and longitudes

summed_dict = {} #dictionary for sums of each grid at each time

#loop through each event
print(f'\n{'='*120}\nsuperposition\n{'='*120}\n{event}')
for event, data_set in df_dictionary.items():
    print(f'\n{'='*120}\nnext event\n{'='*120}\n{event}')

    event_latitude = event_lats[number]
    event_longitude = event_lons[number] #- diff
    print('substorm latitude:', event_latitude)
    print('substorm longitude:', event_longitude, '\n')
    
    #loop through each time in each event
    time_key = 0
    for time, group in data_set.groupby('time'):
        group = group.reset_index(drop = True) #reset index of each group for looping
        
        print(time)
        time_key += 1 #change the time key for adding event grid into dictionary
        index = 0 #index for event grid for every time

        counts = np.zeros(master_index) #empty counts grid
        brightness_grid = np.zeros(master_index) #empty brightness grid
        velocity_grid = np.zeros(master_index) #empty velocity grid
        velx_grid = np.zeros(master_index)
        vely_grid = np.zeros(master_index)
        covariance_grid = np.zeros(master_index) #empty covariance grid
        
        #loop through each grid cell for each time in each event
        for i in range(len(group)):
            
            #calculate upper and lower left latitude corners
            dlat = 1
            dtor = np.pi/180
            lat_ul = float(group['center latitude'][i]) + dlat/2
            lat_ll = float(group['center latitude'][i]) - dlat/2
            
            event_lat = lat_ll
            dlon0 = dlat/np.cos(event_lat * dtor) #initial change in longitude at lowest latitude
            nlons = int(np.round(360/dlon0))
            event_dlon = 360/nlons
            #print('event grid box delta lon:', event_dlon)
            
            lon_lr = float(group['center longitude'][i]) + event_dlon/2
            lon_ll = float(group['center longitude'][i]) - event_dlon/2
            
            brightness_value = float(group['brightness'][i]) #get brightness value
            velocity_value = float(group['velocity'][i]) #get velocity magnitude
            velx_value = float(group['velx'][i]) #get velocity value in the x direction
            vely_value = float(group['vely'][i]) #get velocity value in the y direction
            covariance_value = float(group['covariance'][i]) #get covariance value
            
            event_lat_dist = float(group['center latitude'][i]) - event_latitude
            if lon_ll < 180:
                event_lon_dist = float(group['center longitude'][i]) - (event_longitude + diff)
                
            else:
                event_lon_dist = float(group['center longitude'][i]) - event_longitude
            
            #if event_lon_dist > 180:
                #event_lon_dist = event_lon_dist - 360
            
            #loop through master grid cells to map old grid cell to master grid
            index += 1
            for j in range(master_index):
                
                #get dlon for each latitude in master grid
                master_lat = master_lat_ll[j]
                dlon0 = dlat/np.cos(master_lat * dtor)
                nlons = int(np.round(360/dlon0))
                master_dlon = 360/nlons
                
                master_lat_center = (master_lat_ll[j] + master_lat_ul[j])/2 #center latitude of current master grid cell
                master_lon_center = (master_lon_ll[j] + master_lon_lr[j])/2 #center longitutde of current master grid cell
                
                master_lat_dist = master_lat_center - avg_lat #distance of current grid cell center from the average event latitude
                master_lon_dist = master_lon_center - master_centerlon #distance of current grid cell ffrom the center longitude
                
                #if master_lon_dist > 180:
                    #master_lon_dist = master_lon_dist - 360

                if event_lat_dist == master_lat_dist: #check if the latitude lines align
                    if master_lon_dist - master_dlon/2 < event_lon_dist < master_lon_dist + master_dlon/2: #check if event grid maps into master grid

                        counts[j] += 1 #add counts for averaging
                        brightness_grid[j] += brightness_value
                        velocity_grid[j] += velocity_value
                        velx_grid[j] += velx_value
                        vely_grid[j] += vely_value
                        covariance_grid[j] += covariance_value

                        #print(brightness_value, velocity_value, covariance_value)
                        #find percentage of event grid cell in each master grid cell

                        # if event_lon_dist < master_lon_dist: #checks if event cell center is to the left of master cell center
                        #     term1 = event_lon_dist + event_dlon/2
                        #     term2 = master_lon_dist - master_dlon/2
                        #     percentage1 = ((term1 - term2)/event_dlon) #percentage of event grid cell value that goes into current master grid cell
                        #     percentage2 = 1-percentage1 #percentage of event grid cell value that goes into pervious master grid cell

                        #     #determine if there is a cell in the same latitude before the current one                            
                        #     try:
                        #         prev_lat_center = (master_lat_ll[j-1] + master_lat_ul[j-1])/2
                        #         prev_lat_dist = prev_lat_center - avg_lat

                        #         #add percentages to current and previous master grid cells if possible
                        #         if prev_lat_dist == master_lat_dist:
                        #             counts[j] += 1
                        #             counts[j-1] += 1
                        #             brightness_grid[j] += brightness_value * percentage1
                        #             brightness_grid[j-1] += brightness_value * percentage2

                        #             velocity_grid[j] += velocity_value * percentage1
                        #             velocity_grid[j-1] += velocity_value * percentage2

                        #             covariance_grid[j] += covariance_value * percentage1
                        #             covariance_grid[j-1] += covariance_value * percentage2

                        #         #add percentage to only current master grid cell
                        #         else:
                        #             counts[j] += 1
                        #             brightness_grid[j] += brightness_value * percentage1
                        #             velocity_grid[j] += velocity_value * percentage1
                        #             covariance_grid[j] += covariance_value * percentage1

                        #     except:
                        #         continue
                            
                        # if master_lon_dist < event_lon_dist: #checks if event cell center is to the right of master cell center
                        #     term1 = master_lon_dist + master_dlon/2
                        #     term2 = event_lon_dist - event_dlon/2
                        #     percentage1 = ((term1 - term2)/event_dlon) #percentage of event grid cell that goes into current master grid cell
                        #     percentage2 = 1-percentage1 #percentage of event grid cell value that goes into next master grid cell

                        #     #determine if there is a cell in the same latitude after the current one
                        #     try:
                        #         next_lat_center = (master_lat_ll[j+1] + master_lat_ul[j+1])/2
                        #         next_lat_dist = next_lat_center - avg_lat

                        #         #add percentages to current and next master grid cells if possible
                        #         if next_lat_dist == master_lat_dist:
                        #             counts[j] += 1
                        #             counts[j+1] += 1
                        #             brightness_grid[j] += brightness_value * percentage1
                        #             brightness_grid[j+1] += brightness_value * percentage2
                                    
                        #             velocity_grid[j] += velocity_value * percentage1
                        #             velocity_grid[j+1] += velocity_value * percentage2

                        #             covariance_grid[j] += covariance_value * percentage1
                        #             covariance_grid[j+1] += covariance_value * percentage2

                        #         #add percentage to only current master grid cell
                        #         else:
                        #             counts[j] += 1
                        #             brightness_grid[j] += brightness_value * percentage1
                        #             velocity_grid[j] += velocity_value * percentage1
                        #             covariance_grid[j] += covariance_value * percentage1

                        #     except:
                        #         continue

                        # break
            
        #create a dataframe for the values from the current time
        current_data = pd.DataFrame({
            'brightness': brightness_grid,
            'velocity': velocity_grid,
            'velx': velx_grid,
            'vely': vely_grid,
            'covariance': covariance_grid,
            'counts': counts
        })

        #add dataframe to dictionary based on key
        try:
            if number == 0: #check if the current event is the first event
                summed_dict[time_key] = current_data

            else:
                summed_dict[time_key] = summed_dict[time_key].add(current_data, fill_value = 0)

        except:
            continue

    number += 1 #move to the next event index

#%%average values across the dictionary based on the counts
for key, df in summed_dict.items():
    brightness_avg = df['brightness'].div(df['counts'])
    velocity_avg = df['velocity'].div(df['counts'])
    velx_avg = df['velx'].div(df['counts'])
    vely_avg = df['vely'].div(df['counts'])
    covariance_avg = df['covariance'].div(df['counts'])
    
    df = pd.DataFrame({
            'brightness': brightness_avg,
            'velocity': velocity_avg,
            'velx': velx_avg,
            'vely': vely_avg,
            'covariance': covariance_avg,
    })

    summed_dict[key] = df
#print(summed_dict)

#==================================================================================================
#==================================================================================================
#create line plots if command is called
if line_plot:
    for key, array in summed_dict.items():
        brightness_values = array['brightness']
        velocity_values = array['velocity']
        velx_values = array['velx']
        vely_values = array['vely']
        covariance_values = array['covariance']

        #print(brightness_values)

        for k in range(len(brightness_values)):
            #get values at substorm onset
            if k == 128:
                #print('creating line plots')
                onset_bright.append(brightness_values[k])
                onset_vel.append(velocity_values[k])
                onset_velx.append(velx_values[k])
                onset_vely.append(vely_values[k])
                onset_cov.append(covariance_values[k])

            #get values poleward of substorm onset
            if k == 172:
                pole_bright.append(brightness_values[k])
                pole_vel.append(velocity_values[k])
                pole_velx.append(velx_values[k])
                pole_vely.append(vely_values[k])
                pole_cov.append(covariance_values[k])

            #get values equatorward of substorm onset
            if k == 78:
                eq_bright.append(brightness_values[k])
                eq_vel.append(velocity_values[k])
                eq_velx.append(velx_values[k])
                eq_vely.append(vely_values[k])
                eq_cov.append(covariance_values[k])

            #get values west of substorm onset
            if k == 142:
                west_bright.append(brightness_values[k])
                west_vel.append(velocity_values[k])
                west_velx.append(velx_values[k])
                west_vely.append(vely_values[k])
                west_cov.append(covariance_values[k])

            #get values westward and equatorward of substorm onset
            if k == 71:
                west_eq_bright.append(brightness_values[k])
                west_eq_vel.append(velocity_values[k])
                west_eq_velx.append(velx_values[k])
                west_eq_vely.append(vely_values[k])
                west_eq_cov.append(covariance_values[k])
    
    #==============================================================================================
    #==============================================================================================
    #plot values at substorm onset
    x = range(len(onset_bright))
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    #brightness axis
    axs[0].plot(x, onset_bright, color='tab:blue', marker='o')
    axs[0].set_ylabel('Brightness')
    axs[0].set_ylim(0, 10000)
    axs[0].set_title('Values at Substorm Onset')

    #velocity axis
    axs[1].plot(x, onset_vel, color='tab:orange', marker='s')
    axs[1].set_ylabel('Velocity')
    axs[1].set_ylim(0, 800)

    #covairance axis
    axs[2].plot(x, onset_cov, color='tab:green', marker='^')
    axs[2].set_ylabel('Covariance')
    axs[2].set_ylim(-350000, 350000)
    axs[2].set_xlabel('Event Index')

    plt.tight_layout()
    plt.savefig('lineplots/lineplot_onset.png')
    plt.close()

    #==============================================================================================
    #==============================================================================================
    #plot values poleward of substorm onset
    x = range(len(pole_bright))
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    #brightness axis
    axs[0].plot(x, pole_bright, color='tab:blue', marker='o')
    axs[0].set_ylabel('Brightness')
    axs[0].set_ylim(0, 10000)
    axs[0].set_title('Values Poleward of Substorm Onset')

    #velocity axis
    axs[1].plot(x, pole_vel, color='tab:orange', marker='s')
    axs[1].set_ylabel('Velocity')
    axs[1].set_ylim(0, 800)

    #covairance axis
    axs[2].plot(x, pole_cov, color='tab:green', marker='^')
    axs[2].set_ylabel('Covariance')
    axs[2].set_ylim(-350000, 350000)
    axs[2].set_xlabel('Event Index')

    plt.tight_layout()
    plt.savefig('lineplots/lineplot_poleward.png')
    plt.close()

    #==============================================================================================
    #==============================================================================================
    #plot values equatorward of substorm onset
    x = range(len(eq_bright))
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    #brightness axis
    axs[0].plot(x, eq_bright, color='tab:blue', marker='o')
    axs[0].set_ylabel('Brightness')
    axs[0].set_ylim(0, 10000)
    axs[0].set_title('Values Equatorward of Substorm Onset')

    #velocity axis
    axs[1].plot(x, eq_vel, color='tab:orange', marker='s')
    axs[1].set_ylabel('Velocity')
    axs[1].set_ylim(0, 800)

    #covairance axis
    axs[2].plot(x, eq_cov, color='tab:green', marker='^')
    axs[2].set_ylabel('Covariance')
    axs[2].set_ylim(-350000, 350000)
    axs[2].set_xlabel('Event Index')

    plt.tight_layout()
    plt.savefig('lineplots/lineplot_equatorward.png')
    plt.close()

    #==============================================================================================
    #==============================================================================================
    #plot values to the west of substorm onset
    x = range(len(west_bright))
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    #brightness axis
    axs[0].plot(x, west_bright, color='tab:blue', marker='o')
    axs[0].set_ylabel('Brightness')
    axs[0].set_ylim(0, 10000)
    axs[0].set_title('Values West of Substorm Onset')

    #velocity axis
    axs[1].plot(x, west_vel, color='tab:orange', marker='s')
    axs[1].set_ylabel('Velocity')
    axs[1].set_ylim(0, 800)

    #covairance axis
    axs[2].plot(x, west_cov, color='tab:green', marker='^')
    axs[2].set_ylabel('Covariance')
    axs[2].set_ylim(-350000, 350000)
    axs[2].set_xlabel('Event Index')

    plt.tight_layout()
    plt.savefig('lineplots/lineplot_west.png')
    plt.close()

    #==============================================================================================
    #==============================================================================================
    #plot values to the west and equatorward of substorm onset
    x = range(len(west_eq_bright))
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    #brightness axis
    axs[0].plot(x, west_eq_bright, color='tab:blue', marker='o')
    axs[0].set_ylabel('Brightness')
    axs[0].set_ylim(0, 10000)
    axs[0].set_title('Values West and Equatorward of Substorm Onset')

    #velocity axis
    axs[1].plot(x, west_eq_vel, color='tab:orange', marker='s')
    axs[1].set_ylabel('Velocity')
    axs[1].set_ylim(0, 800)

    #covairance axis
    axs[2].plot(x, west_eq_cov, color='tab:green', marker='^')
    axs[2].set_ylabel('Covariance')
    axs[2].set_ylim(-350000, 350000)
    axs[2].set_xlabel('Event Index')

    plt.tight_layout()
    plt.savefig('lineplots/lineplot_west_equatorward.png')
    plt.close()

#==================================================================================================
#==================================================================================================
#%%create and plot superposed grid at each time
#set brightness normalization
bright_colormap = mpl.colormaps['rainbow'] #get colormap object from matplotlib colormap
bright_norm = Normalize(vmax = 5000, vmin = 0)

#set velocity normalization
vel_colormap = mpl.colormaps['plasma'] #get colormap object from matplotlib colormap
vel_norm = Normalize(vmax = 800, vmin = 0)

#set covariance normalization
cov_colormap = mpl.colormaps['RdYlGn'] #get colormap object from matplotlib colormap
cov_norm = Normalize(vmax = 350000, vmin = -350000)

center_lats = []
center_lons = []
patches = []

#loop through time keys
for key, df in summed_dict.items():
    collection = []

    if key == 1:
        print(f'\nplotting time frame: {key:02}')

    else:
        print(f'plotting time frame: {key:02}')

    #place grid cells from selected region into patches for plotting
    bright_patches = []
    vel_patches = []
    cov_patches = []

    brightness = df['brightness']
    velocity = df['velocity']
    velx = np.array(df['velx'])
    vely = np.array(df['vely'])
    covariance = df['covariance']

    color = (0, 1, 1)

    #loop through master grid cells
    for lat1, lat2, lon1, lon2, bright, vel, cov in zip(master_lat_ll, master_lat_ul, master_lon_ll, master_lon_lr, brightness, velocity, covariance):

        x = lon1
        y = lat1

        width = lon2 - lon1
        height = lat2 - lat1

        bright_color = bright_colormap(bright_norm(bright))
        bright_patches.append(Rectangle((x, y), width, height, color = bright_color, ec = None))

        vel_color = vel_colormap(vel_norm(vel))
        vel_patches.append(Rectangle((x, y), width, height, color = vel_color, ec = None))

        cov_color = cov_colormap(cov_norm(cov))
        cov_patches.append(Rectangle((x, y), width, height, color = cov_color, ec = None))

        #create figure showing the grid indicies for the master grid
        if key == 16:
            cell_lat = (lat2 + lat1)/2
            cell_lon = (lon2 + lon1)/2
            center_lats.append(cell_lat)
            center_lons.append(cell_lon)

            patches.append(Rectangle((x, y), width, height, color = color, ec = None))

    P = PatchCollection(patches, transform = ccrs.PlateCarree(), match_original = True)

    BrightP = PatchCollection(bright_patches, transform = ccrs.PlateCarree(), match_original = True)
    VelP = PatchCollection(vel_patches, transform = ccrs.PlateCarree(), match_original = True)
    CovP = PatchCollection(cov_patches, transform = ccrs.PlateCarree(), match_original = True)

    collection.append(VelP)
    collection.append(BrightP)
    collection.append(CovP)

    #%%create figure to plot patches onto, this is the grid where the average values will be placed into
    latitude_lines = np.linspace(southlat, northlat + 1, 10)
    latitudes = np.linspace(southlat - 1, northlat + 2, 100) #list of latitude points to plot a longitude line along

    dlat = 1
    dtor = np.pi/180
    dlon0 = dlat/np.cos(avg_lat * dtor) #initial change in longitude at lowest latitude
    nlons = int(np.round(360/dlon0))
    dlon = 360/nlons

    if westlon > eastlon:
        westlon = westlon - 360

    longitudes = np.linspace(westlon + 5, eastlon + dlon, 100) #list of longitude points to plot a longitude line along

    fig, axes = plt.subplots(3, 1, figsize = (10, 15), dpi = 200, subplot_kw = {'projection': ccrs.AzimuthalEquidistant(
        central_longitude = master_centerlon - diff, central_latitude = 65, false_easting = 0.0, false_northing = 0.0, globe = None)})
    
    #add colorbars for brightness, velocity, and covariance
    #velocity colormap
    if vectors:
        print('creating velocity plot using vectors')

        #get speed for colormap
        speed = np.sqrt(velx**2 + vely**2)

        q = axes[0].quiver(
            (master_lon_ll + master_lon_lr)/2, (master_lat_ll + master_lat_ul)/2,
            velx, vely,
            speed,
            cmap = 'gnuplot',
            norm = vel_norm,
            scale = 1000,
            scale_units='inches',
            width = 0.003,
            transform = ccrs.PlateCarree()
        )

        cbar_ax = fig.add_axes([0.88, 0.86, .02, .12])
        cbar = fig.colorbar(cm.ScalarMappable(norm = vel_norm, cmap = 'gnuplot'), cax = cbar_ax, shrink = 0.5)
        cbar.ax.set_ylabel('Plasma Velocity (m/s)')

    else:
        cbar_ax = fig.add_axes([0, 0, .1, .1]) #create new axis for colorbar; left, bottom, width, and height of axis (maximum of 1 for each value)
        cbar_ax.set_position([0.88, 0.86, .02, .12]) #eft, bottom, width, and height; set position of colorbar
    
        cbar = fig.colorbar(cm.ScalarMappable(norm = vel_norm, cmap = 'plasma'), cax = cbar_ax, shrink = 0.5)
        cbar.ax.set_ylabel('Plasma Velocity Magnitude (m/s)')

    #brightness colormap
    cbar_ax = fig.add_axes([0, 0, .1, .1]) #create new axis for colorbar; left, bottom, width, and height of axis (maximum of 1 for each value)
    cbar_ax.set_position([0.88, 0.53, .02, .12]) #eft, bottom, width, and height; set position of colorbar
    
    cbar = fig.colorbar(cm.ScalarMappable(norm = bright_norm, cmap = 'rainbow'), cax = cbar_ax, shrink = 0.5)
    cbar.ax.set_ylabel('Auroral Brightness (Unitless)')
    
    #covariance colormap
    cbar_ax = fig.add_axes([0, 0, .1, .1]) #create new axis for colorbar; left, bottom, width, and height of axis (maximum of 1 for each value)
    cbar_ax.set_position([0.88, 0.20, .02, .12]) #eft, bottom, width, and height; set position of colorbar
    
    cbar = fig.colorbar(cm.ScalarMappable(norm = cov_norm, cmap = 'RdYlGn'), cax = cbar_ax, shrink = 0.5)
    cbar.ax.set_ylabel('Covariance (Unitless)')
    ticks = cbar.get_ticks()
    cbar.ax.set_yticklabels([f'{tick:.0e}' for tick in ticks])

    #plot annotations on each subplot
    for ax in axes:
        ax.set_adjustable('datalim')
        ax.set_extent((westlon, eastlon + dlon, min_lat, max_lat), ccrs.PlateCarree())  #set projection limits
        gl = ax.gridlines()  #add latitude/longitude grid lines
        gl.ylocator = plt.FixedLocator(latitude_lines) #specify specific lines of latitude to plot
        ax.set_clip_on(True)
        
        ax.set_facecolor('grey')
        
        #plot lines of latitude and longitude aligned with substorm onset
        ax.plot(longitudes - 1, [avg_lat] * len(longitudes),
                transform = ccrs.PlateCarree(), color = 'black', linewidth = 5, linestyle = '--') #line along substorm onset latitude
        
        ax.plot([master_centerlon] * len(latitudes), latitudes,
                transform = ccrs.PlateCarree(), color = 'black', linewidth = 5, linestyle = '--') #line along substorm onset longitude
        
        ax.plot([eastlon] * len(latitudes), latitudes,
                transform = ccrs.PlateCarree(), color = 'black', linewidth = 2.5, linestyle = 'dashdot') #line along eastern longitude boundary
    
        ax.plot([westlon] * len(latitudes), latitudes,
                transform = ccrs.PlateCarree(), color = 'black', linewidth = 2.5, linestyle = 'dashdot') #line along western longitude boundary
    
        #add annotation for time and line lables
        xpos, ypos = 0.02, 0.93
        if key == 16:
            ax.annotate(f'Frame {key:02} (substorm onset)', (xpos, ypos), xycoords = 'axes fraction', fontsize = 16,
                        bbox = dict(facecolor = 'white', edgecolor = 'none', boxstyle = 'round, pad = 0.3'))

        else:
            ax.annotate(f'Frame {key:02}', (xpos, ypos), xycoords = 'axes fraction', fontsize = 16,
                        bbox = dict(facecolor = 'white', edgecolor = 'none', boxstyle = 'round, pad = 0.3'))
        
        ax.annotate(f'Substorm Onset\nMLT',
                    xy = (master_centerlon, avg_lat + 6),
                    xycoords = ccrs.PlateCarree()._as_mpl_transform(ax),
                    fontsize = 11,
                    bbox = dict(facecolor = 'white', edgecolor = 'none', boxstyle = 'round, pad = 0.3'),
                    ha = 'center'
                    )
        
        xpos, ypos = 0.09, 0.75
        ax.annotate(f'Average Substorm Onset \nLatitude ({round(avg_lat, 2)})', (xpos, ypos), xycoords = 'axes fraction', fontsize = 11,
                    bbox = dict(facecolor = 'white', edgecolor = 'none', boxstyle = 'round, pad = 0.3'), ha = 'center')

    #plot patch collections onto each subplot
    axes[1].add_collection(BrightP)
    axes[2].add_collection(CovP)

    if not vectors:
        axes[0].add_collection(VelP)

    plt.tight_layout()
    plt.savefig(f'superposition_grid_{key:02}.png')
    plt.close()

#==================================================================================================
#==================================================================================================
#create plot showing master grid indicies at substorm onset
fig = plt.figure(figsize = (10, 10), dpi = 200)

#define coordinate reference system
crs = ccrs.AzimuthalEquidistant(central_longitude = master_centerlon - diff,
                                central_latitude = avg_lat,
                                false_easting = 0.0,
                                false_northing = 0.0,
                                globe = None)

ax = fig.add_subplot(1, 1, 1, projection = crs)
fig.set_size_inches(10, 6)

ax.set_adjustable('datalim')
ax.set_extent((westlon, eastlon, southlat, northlat), ccrs.PlateCarree()) #setting limits of projection to be northern hemisphere
ax.set_facecolor('grey')
ax.set_clip_on(True)

#ax.gridlines(ylocs = latitudes) #add latitude and longitude grid lines

ax.add_collection(P)

#add index numbers over each grid
centerlats = np.array(center_lats)
centerlons = np.array(center_lons)
for i in range(master_index):
    ax.text(x = centerlons[i], y = centerlats[i],
            s = str(i), ha = 'center', va = 'center',
            fontsize = 6, color = 'black', weight = 'bold',
            transform = ccrs.PlateCarree())
    
#plt.savefig('lineplots/master_index.png')

print(f'\nmaster grid center longitude: {master_centerlon}')
print(f'eastlon: {eastlon}')
print(f'westlon: {westlon}')
print(f'master grid center MLT: {center_MLT}')
print(f'difference between center longitude and 0 longitude line: {diff}')
print(f'master grid index: {master_index}')

print(f'\n{'='*120}\nfigures made\n{'='*120}')