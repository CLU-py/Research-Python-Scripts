#!/usr/bin/env python3

#import required modules
import cdflib
import aacgmv2
import numpy as np
import pandas as pd
import datetime as dt

from collections import defaultdict

#%%set satellites and other common variables
sats = ['A', 'B', 'C']
date = '20140321'
root_path = 'E:\Python Scripts\Research/'

#set time for aacgm
year = int(date[0:4])
month = int(date[4:6])
day = int(date[7:9])
dtime = dt.datetime(year, month, day, 0, 0, 0)

#create three empty dataframes
dfs = [pd.DataFrame() for _ in range(3)]

#create empty dictionaries for position and velocity values
swarm_pos = defaultdict(list) #empty dictionary for satellite positions
swarm_vels = defaultdict(list)  #empty dictionary for velocities

#%%set paths and open files
for i in range(len(sats)):
    print(f'opening sat {sats[i]}')
    cdf_path = f'{root_path}SW_EXPT_EFI{sats[i]}_TCT02_{date}T000000_{date}T235959_0401/'
    cdf_file = f'SW_EXPT_EFI{sats[i]}_TCT02_{date}T000000_{date}T235959_0401.cdf' #2Hz file

    cdf_open = cdflib.CDF(cdf_path + cdf_file)
    #print(cdf_open.cdf_info()) #print variables of cdf file

    #get variables
    timestamps = cdf_open.varget('Timestamp') #times are seconds from 1 Jan 2000 00:00:00 UT
    datetimes_array = cdflib.cdfepoch.to_datetime(timestamps)
    datetimes = datetimes_array.astype('datetime64[ms]').astype(object) #convert to datetime object

    #set geographic points
    latitude = cdf_open.varget('latitude') #ITRF spherical latitude derived from L1B Medium Orbit Determination (MOD), geographic latitude
    longitude = cdf_open.varget('longitude') #ITRF spherical longitude derived from L1B MOD, geographic longitude
    radius = cdf_open.varget('radius') #ITRF spherical radius derived from L1B MOD
    altitude = radius / 1000 - 6371 #convert meters to kilometers

    #convert geographic points to aacgm coordinates
    aacgm_coords = np.array(aacgmv2.get_aacgm_coord_arr(latitude, longitude, altitude, dtime))
    #aacgm_lat = aacgm_coords[0, :]
    #aacgm_lon = aacgm_coords[1, :]
    #aacgm_mlt = aacgm_coords[2, :]

    v_ion_H = cdf_open.varget('Vixh')
    v_ion_V = cdf_open.varget('Vixv')
    v_ion_y = cdf_open.varget('Viy')
    v_ion_z = cdf_open.varget('Viz')

    qual_flag = cdf_open.varget('Quality_flags')
    cal_flag = cdf_open.varget('Calibration_flags')

    #add variables to a dataframe
    dfs[i] = pd.DataFrame({
        'latitude': latitude,
        'longitude': longitude,
        'altitude': altitude,
        'aacgm_lat': aacgm_coords[0, :],
        'aacgm_lon': aacgm_coords[1, :],
        'mlt': aacgm_coords[2, :],
        'v_ion_H': v_ion_H,
        'v_ion_V': v_ion_V,
        'v_ion_y': v_ion_y,
        'v_ion_z': v_ion_z,
        'quality_flag': qual_flag,
        'calibration_flag': cal_flag,
    }, index = datetimes)
    
    dfs[i] = dfs[i][dfs[i].index.microsecond == 231000]
    dfs[i].index = dfs[i].index.floor('s')
    
    
    
dfs = dict(zip(sats, dfs))

#%%loop through and append the values to the specific dictionary
time = dfs['A'].index.to_numpy().astype('datetime64[ms]').astype(object)
for sat in sats:
    swarm_pos[sat].append(time)
    swarm_pos[sat].append(dfs[sat]['aacgm_lat'].to_numpy())
    swarm_pos[sat].append(dfs[sat]['aacgm_lon'].to_numpy())
    
    swarm_vels[sat].append(time)
    swarm_vels[sat].append(dfs[sat]['aacgm_lat'].to_numpy())
    swarm_vels[sat].append(dfs[sat]['aacgm_lon'].to_numpy())
    swarm_vels[sat].append(dfs[sat]['v_ion_H'].to_numpy())
    
    
    
    
    
    
    
    
    
    
    
    