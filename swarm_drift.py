#!/usr/bin/env python3

#import required modules
import cdflib
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

#%%link for TII product description
#https://swarmhandbook.earth.esa.int/catalogue/sw_efix_tii1b

#%%set paths and open files
cdf_path = 'E:\Python Scripts\Research\SW_PREL_EFIA_TII1B_20131225T000000_20131226T000000_0103/'
cdf_file = 'SW_PREL_EFIA_TII1B_20131225T000000_20131226T000000_0103.cdf'

cdf_open = cdflib.CDF(cdf_path + cdf_file)
print(cdf_open.cdf_info()) #print variables of cdf file

#get variables
timestamps = cdf_open.varget('Timestamp') #times are seconds from 1 Jan 2000 00:00:00 UT
epoch = datetime(2000, 1, 1, 0, 0, 0) #reference epoch
datetimes = np.array([epoch + timedelta(seconds = t) for t in timestamps]) #convert to array of datetime objects

latitude = cdf_open.varget('latitude') #ITRF spherical latitude derived from L1B Medium Orbit Determination (MOD), geographic latitude
longitude = cdf_open.varget('longitude') #ITRF spherical longitude derived from L1B MOD, geographic longitude
radius = cdf_open.varget('radius') #ITRF spherical radius derived from L1B MOD
altitude = radius / 1000 - 6371 #convert meters to kilometers

v_ion = cdf_open.varget('v_ion') #ion velocity vector in North East and Centre (NEC) frame
v_ion_H = cdf_open.varget('v_ion_H') #horizontal sensor ion velocity in TII coordinates x and y
v_ion_V = cdf_open.varget('v_ion_V') #vertical sensor ion velocity in TII coordinates x and y

flags_TII = cdf_open.varget('Flags_TII') #TII quality flag
flags_platform = cdf_open.varget('Flags_Platform') #satellite platform flag

#add variables to a dataframe
df_A = pd.DataFrame({
    'latitude': latitude,
    'longitude': longitude,
    'altitude': altitude,
    'v_ion_N': v_ion[:, 0],
    'v_ion_E': v_ion[:, 1],
    'v_ion_C': v_ion[:, 2],
    'flags-TII': flags_TII,
    'flags_platform': flags_platform
}, index = datetimes)