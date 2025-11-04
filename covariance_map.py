#!/usr/bin/env python3

#THIS SCRIPT CREATES A HEMISPHERICAL MAP PLOTTING COVARIANCE AT EACH MLT FOR THE GIVEN TIME PERIOD
#TO CALL THIS SCRIPT: filename starttime endtime (vel_out 201403210400 201403210600)
#IF SCRIPT IS NOT EXECUTABLE IN TERNIMAL, RUN chmod +x {filename}.py

#LENGTH OF TIME INTERVAL IMPACTS COVARIANCE DUE TO AVERAGE VELOCITY AND BRIGHTNESS BEING CALCULATED ALONG ACROSS THE ENTIRE INTERVAL FOR EACH GRID CELL

#TO RUN PROCESS IN SCREEN MODE: 'screen' TO START SCREEN SESSION, crtl + a (release), then d TO DETACH THE PROCESS/SCREEN
#TO RESUME DETACHED PROCESS: screen -r IN TERMINAL

#TO SEE PROCESSES RUNNING IN TERMINAL: 'top' IN COMMAND LINE

#==========================================================================================================================================
#==========================================================================================================================================
#import required modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.dates as mdates
import matplotlib.cm as cm
import argparse
import os
import glob
import datetime as dt
import cartopy .crs as ccrs
import aacgm
import read_df_record

from matplotlib.colors import Normalize
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import cdflib
from cdflib import cdfread
from cdflib.xarray import cdf_to_xarray

from df_vel_class import df_vel
from date_strings import cnv_datetimestr_dtlist, cnv_datetimestr_datetime
from map_grid import make_grid

#==========================================================================================================================================
#==========================================================================================================================================
#add commands to call in terminal when calling script; seae top of script for command order
parser = argparseparser = argparse.ArgumentParser(description='sector_gdf_vel argument parser')
parser.add_argument('fname', type = str) #file name of vel_out file
parser.add_argument('sttime', type = str) #start time argument; YYYmmddHHMM
parser.add_argument('ndtime', type = str) #end time argument; YYYmmddHHMM
parser.add_argument('--meridians', type = float, nargs = "+", default = None)

args = parser.parse_args()
fname = args.fname
sttime = args.sttime
ndtime = args.ndtime
meridians = args.meridians

#==========================================================================================================================================
#==========================================================================================================================================
#define function to plot meridian lines
def cnv_mlt_label(mlt):
    mlt_hour=int(mlt)
    mlt_min=int(60*(mlt-mlt_hour))

    return(str(mlt_hour).rjust(2,'0')+str(mlt_min).rjust(2,'0')+" MLT")

#==========================================================================================================================================
#==========================================================================================================================================
#define function to read ASI cdf file
def cdf_read_file(cdf_f,site,start_time,end_time):
    var='thg_asf_'+site
    sttime = [start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second, 0]
    ndtime = [end_time.year, end_time.month, end_time.day, end_time.hour, end_time.minute, end_time.second+delta_t, 0]
    img = cdf_f.varget(var, starttime = sttime, endtime = ndtime)
    if (not isinstance(img, (list, tuple, np.ndarray))) or (img is None):
        return(-1)

    return(img) 

#==========================================================================================================================================
#==========================================================================================================================================
#define function to get ASI site fov
def thg_asi_fov(site, time, mag = False, alt = 1):
    fov_path = '/import/SUPERDARN/themis_asi/fovMapping/' #path to fov data
    fname = glob.glob(fov_path + 'thg_l2_asc_' + site + '*') #name of fov file
    #print(fname[0])

    cal_f = cdfread.CDF(fname[0])

    sttime = [time.year, time.month, time.day, time.hour, time.minute, time.second, 0] #start time
    #print('mag = ',mag)
    #print('Start time:', sttime)
    if mag:
        sta_lat = cal_f.varget('thg_asc_' + site + '_mlat', starttime = sttime) #station latitude
        sta_lon = cal_f.varget('thg_asc_' + site + '_mlon', starttime = sttime) #station longitude
        #print('Station location:', sta_lat, sta_lon)
        if (sta_lat is None):
            return(-1)

        lats = cal_f.varget('thg_asf_' + site + '_mlat', starttime = sttime)[0] #latitudes in asi data
        lons = cal_f.varget('thg_asf_' + site + '_mlon', starttime = sttime)[0] #longitudes in asi data
    else:
        sta_lat = cal_f.varget('thg_asc_' + site + '_glat', starttime = sttime) #station latitude
        sta_lon = cal_f.varget('thg_asc_' + site + '_glon', starttime = sttime) #station longitude
        #print('Station location:', sta_lat, sta_lon)
        if (sta_lat is None):
            return(-1)

        try:
            lats = cal_f.varget('thg_asf_' + site + '_glat',starttime = sttime)[0] #latitudes in asi data
        except:
            return(-1)
        try:
            lons = cal_f.varget('thg_asf_'+site + '_glon', starttime = sttime)[0] #longitudes in asi data
        except:
            return(-1)

    lats = np.delete(lats, 256, 0)
    lats = np.delete(lats, 256, 1)
    lons = np.delete(lons, 256, 0)
    lons = np.delete(lons, 256, 1)

    elev=cal_f.varget('thg_asf_'+site+'_elev',starttime=sttime)[0] #station elevation
    return(sta_lat,sta_lon,lats,lons,elev)

mag = True #if True, script is using magnetic time; keep true for script to run properly

#==========================================================================================================================================
#==========================================================================================================================================
#create map grid and empty dictionaries for velocity and brightness values
grid = make_grid()

lat_ll = grid['lat_ll'] #list of lower left latitudes
lat_ul = grid['lat_ul'] #list of upper left latitudes

lon_ll = grid['lon_ll'] #list of lower left longitudes
lon_lr = grid['lon_lr'] #list of lower right longitudes

grid_points = len(grid['index'])

vel_grids = {} #empty dictionary for each velocity grid set at each time where radar data was considered
vel_grids_nd = {} #empty dictionary for each velocity grid set at each time where no radar data was considered
bright_grids = {} #empty dictionary for each brightness grid set at each time

#==========================================================================================================================================
#==========================================================================================================================================
#get velocity data at every MLT
print('\nGet Velocity')
print('==================================================================================================================================')
print('==================================================================================================================================\n')
dfVel = df_vel(fname, global_vel = True)

if mag:
    aacgm.set_datetime(dfVel.dstr.yr,dfVel.dstr.mo,dfVel.dstr.dy,int(dfVel.dstr.hr),int(dfVel.dstr.mt),0)

    for j in range(dfVel.dstr.num):
        pos=aacgm.convert(dfVel.dstr.vec_lat[j],dfVel.dstr.vec_lon[j],300,0)
        #print(pos)
        dfVel.dstr.vec_lat[j]=pos[0]
        dfVel.dstr.vec_lon[j]=pos[1]

dtor=np.pi/180
inds=[]
dlon = .75 #maybe have to increase this
for idx,lon in enumerate(dfVel.dstr.vec_lon):
    lat = dfVel.dstr.vec_lat[idx] #magnetic or planetary latitude?
    #print('The lat value is:', lat)
    p_lon = aacgm.inv_mlt_convert(dfVel.dstr.yr, dfVel.dstr.mo, dfVel.dstr.dy, int(dfVel.dstr.hr), int(dfVel.dstr.mt), 0, 23) #converts MLT to planetary longitude

    if p_lon<0: p_lon+=360
    #print('The plon is:', p_lon)

    if (lon > p_lon-dlon/(2*np.cos(lat*dtor))) and (lon < p_lon+dlon/(2*np.cos(lat*dtor))):
        inds.append(idx)

lats=dfVel.dstr.vec_lat[inds]

rtimes = []

st_tm = cnv_datetimestr_dtlist(sttime)
start_datetime = dt.datetime(st_tm[0], st_tm[1], st_tm[2], st_tm[3], st_tm[4])

nd_tm = cnv_datetimestr_dtlist(ndtime)
end_datetime = dt.datetime(nd_tm[0], nd_tm[1], nd_tm[2], nd_tm[3], nd_tm[4])

#==========================================================================================================================================
#==========================================================================================================================================
#calculte x and y velocities over the entire hemisphere and map to grid
intervals = 0
while 1:
    rtime = dt.datetime(dfVel.dstr.yr, dfVel.dstr.mo, dfVel.dstr.dy,
                              dfVel.dstr.hr, dfVel.dstr.mt, int(dfVel.dstr.sc))

    #print(intervals)
    print(rtime) #current time velocity data is being taken from
    if rtime < start_datetime:
        dfVel.read_record()
        if not isinstance(dfVel.dstr, read_df_record.record):
            break

        #print('Data not in selected time interval')
        continue

    if rtime > end_datetime:
        break

    rtimes.append(rtime)

    #======================================================================================================================================
    #get velocities
    vx = dfVel.dstr.vx #x-velocity values over entire hemisphere at the given time
    vy = dfVel.dstr.vy #y-velocity values over entire hemisphere at the given time

    v_mag = np.sqrt(vx**2 + vy**2) #velocity magnitude over entire hemisphere

    v_lats = dfVel.dstr.vec_lat #latitude of the velocity vectors
    v_lons = dfVel.dstr.vec_lon #longitude of velocity vectors

    #======================================================================================================================================
    #define empty grid and count arrays
    vel_grid = np.zeros(len(grid['index'])) #empty list for velocity grid where radar data was used
    vel_grid_nd = np.zeros(len(grid['index']))
    vel_count = np.zeros(len(grid['index'])) #empty array to hold a count for the number of values in each bin
    vel_count_nd = np.zeros(len(grid['index'])) #empty array to hold a count for the number of values in each bin

    #==================================================================================================================================
    #get indicies of latitudes and longitudes and place velocity values into grid
    for k in range(len(v_mag)):
        lat_inds = np.asarray((lat_ll < v_lats[k]) & (lat_ul > v_lats[k])).nonzero()
        lon_ind = np.asarray((lon_ll[lat_inds] < v_lons[k]) & (lon_lr[lat_inds] > v_lons[k])).nonzero()

        try:
            #if dfVel.dstr.dcount[k] != 0: #velocity value did not consider data from radars
                #j_indx = lat_inds[0][lon_ind[0][0]]
                #vel_grid_nd[j_indx] += v_mag[k]
                #vel_count_nd[j_indx] += 1

            #else: #velocity value considered data from radars
            j_indx = lat_inds[0][lon_ind[0][0]]
            vel_grid[j_indx] += v_mag[k]
            vel_count[j_indx] += 1
        
        except:
            continue

    nz_vel = vel_count > 0
    vel_grid[nz_vel] /= vel_count[nz_vel]

    #nz_vel_nd = vel_count_nd > 0
    #vel_grid_nd[nz_vel_nd] /= vel_count_nd[nz_vel_nd]

    vel_grids.update({intervals:vel_grid})
    #vel_grids_nd.update({intervals:vel_grid_nd})

    #======================================================================================================================================
    #advance dfVel file to next time 
    dfVel.read_record()
    if not isinstance(dfVel.dstr,read_df_record.record):
        break

    intervals += 1

#print('velocity intervals', intervals)

#==========================================================================================================================================
#==========================================================================================================================================
#get brightness data from every ASI site and at every MLT
print('\nGet Brightness')
print('==================================================================================================================================')
print('==================================================================================================================================')
start_time = cnv_datetimestr_datetime(sttime)
end_time = cnv_datetimestr_datetime(ndtime)

#==========================================================================================================================================
#==========================================================================================================================================
#define change in latitude and longitude and define time intervals
lat_min = 55
lat_max = 80

if mag:
    lon_max = 30.0
    lon_min = -130.0
else:
    lon_max = -20
    lon_min = -180

dlat = .15
dlon = 1*dlat/np.cos((90-(lat_max-lat_min)/2)*np.pi/180.)

delta_t = 3 #timestep between images in seconds

nlat = int((lat_max - lat_min) / dlat) #number of latitudes; 166
nlon = int((lon_max - lon_min) / dlon) #number of longitudes; 230

n_intervals = int(((end_time - start_time).total_seconds())/delta_t) #number total time of intervals

ptimes = np.zeros((nlat,n_intervals))
from df_vel_class import df_vel #dataframe for velocity

#==========================================================================================================================================
#==========================================================================================================================================
#list station sites
#sites = ['rank'] #single asi site
sites = ['kian', 'atha', 'fykn', 'inuv', 'whit', 'gako', 'fsmi', 'fsim', 'gill', 'rank', 'kuuj', 'snkq'] #every asi site
scales = {'atha':2000, 'fsim':4500, 'fykn':2000, 'inuv':2000, 'whit':2500, 'fsmi':5000, 'gako':2000, 'gill':2500, 'rank':4000, 'kian':3000, 'snkq':4000}

n_hours = int(np.rint((end_time - start_time).seconds/3600)) #number of hours in bewtween the start time and end time

#==========================================================================================================================================
#==========================================================================================================================================
#get asi data
datestr = start_time.strftime("%Y%m%d")
data_path = '/import/SUPERDARN/themis_asi/asi_cdfs/'+datestr+'/'

time = start_time
for jt in range(n_intervals):
    time = time + dt.timedelta(seconds = delta_t)
    ptimes[:, jt] = mdates.date2num(time)

#==========================================================================================================================================
#==========================================================================================================================================
#make array for latitudes with a dlat of one degree
latitudes = np.arange(60, 81, 1) #create an array of latitudes with a step of 1

min_lat = np.min(latitudes) #minimum latitude
max_lat = np.max(latitudes) #maximum latitude

#==========================================================================================================================================
#==========================================================================================================================================
#set initial times and intervals
file_time = start_time #time function to read cdf considers
minute_intervals = 30 #number of two minute intervals in a single hour

#==========================================================================================================================================
#==========================================================================================================================================
#set dictionaries and loop through hour intervals
imgs = {}
site_lats = {} #dictionary to hold brightness latitudes for each site
site_lons = {} #dictionary to hold brightness longitudes for each site
site_elev = {}
data_arrays = {} #dictionary to hold data variable for each site

#==========================================================================================================================================
#==========================================================================================================================================
#loop through hours
intervals = 0
for hr in range(n_hours):
    #print('\n')
    #print(file_time)
    image_time = file_time + dt.timedelta(minutes = 1)
    fov_error_sites = []
    file_error_sites = []

    if mag:
        aacgm.set_datetime(file_time.year, file_time.month, file_time.day, file_time.hour, file_time.minute, file_time.second)

    #======================================================================================================================================
    #loop through sites    
    for site in sites:
        print('\n', site)
        
        datestr = file_time.strftime("%Y%m%d%H")
        cdf_name = data_path + 'thg_l1_asf_' + site + '_' + datestr + '_v01.cdf'
        print(hr, cdf_name)
        
        if not os.path.exists(cdf_name):
            print("FILE NOT FOUND: ", cdf_name)
            file_error_sites.append(site)
            continue
        
        cdf_f = cdfread.CDF(cdf_name)
        imgs.update({site:cdf_read_file(cdf_f, site, start_time, end_time)}) #brightnesses in three second intervals for the given hour; each with 256 values

        #==================================================================================================================================
        #get camera latitude, longitude and latitudes, longitudes of camera FOV
        try:
            sta_lat, sta_lon, lats, lons, elev = thg_asi_fov(site, file_time, mag = mag) #latitude and longitude arrays are each 256 values
            
        except:
            print('problem with fov')
            fov_error_sites.append(site)
            continue

        site_lats.update({site:lats}) #add the current sites latitudes to the dictionary
        site_lons.update({site:lons}) #add the current sites longitudes to the dictionary
        site_elev.update({site:elev}) #add the current sites elevations to the dictionary
        
        try:
            data = cdf_to_xarray(cdf_name, to_datetime = True)
        except:
            print("NO DATA: ", cdf_name)
            continue

        data_arrays.update({site:data}) #add the current sites data variable to the dictionary
    
    #======================================================================================================================================
    #loop through two minute intervals
    for i in range(minute_intervals):
        #print(intervals)
        print('\n', image_time)

        bright_grid = np.zeros(len(grid['index'])) #empty array to hold brightness values from each site
        bright_count = np.zeros(len(grid['index'])) #empty array to hold brightness values from each site

        #==================================================================================================================================
        #loop through sites and read data
        b_lats = [] #empty array for brightness latitudes
        b_lons = [] #empty array for brightness longitudes
        b_mag = [] #empty array for brightness magnitude

        for site in sites:
            if site in fov_error_sites:
                print(site, 'FOV error')
                continue

            if site in file_error_sites:
                print(site, 'file error')
                continue
            
            print(site)

            #==============================================================================================================================
            #get the time of each interval for the given hour
            imgs_name = 'thg_asf_'+site
            times_name = imgs_name+'_epoch'
            ns = 1.e-9
            times = data_arrays[site][times_name].values #time of each image for the current site

            n_img = len(imgs[site]) #number of intervals per hour
            
            date_times = []

            #==============================================================================================================================
            #chage times into datetime objects for averaging
            a = []
            for j in range(n_img):
                time_dt = dt.datetime.utcfromtimestamp(times[j].astype(int) * ns) #convert times array into datetime objects; use when SSH'ing
                #time_dt = dt.datetime.fromtimestamp(times[j].astype(int) * ns, dt.UTC) #convert times array into datetime objects; use on wilcox.met
                date_times.append(time_dt.replace(tzinfo = None)) #add the list of datetimes to the array with the proper format

                state = date_times[j] > image_time and date_times[j] < image_time + dt.timedelta(minutes = 2) #determines whether the selected date time is within the current image time range
                a.append(state)

            #==============================================================================================================================
            #average image data for given hour 
            img_avg = np.mean(imgs[site][a], axis = 0)

            #==============================================================================================================================
            #saturate the data from each site
            nrows = site_lats[site].shape[0] #number of rows; 256
            ncols = site_lats[site].shape[1] #number of columns; 256

            dark_image = np.zeros((nrows, ncols))
            try:
                for jr in range(nrows):
                    for jc in range(ncols):
                        dark_image[jr, jc] = np.nanmin(imgs[site][:][jr, jc])

                image_data = img_avg - dark_image
                #image_data = imgs[site][i*40] - dark_image

            except:
                continue

            is_nan = np.all(np.isnan(image_data)) #if True, the entire image data array is full of nans (the current imager has not turned on)

            if is_nan:
                continue

            neglons = site_lons[site] < 0 #find longitudes that are negative 
            site_lons[site][neglons] += 360 #set all negative longitudes to positive for plotting

            #==============================================================================================================================
            #find latitudes and longitudes corresponding to each pixel in the image
            for row in range(nrows):
                for column in range(ncols):
                    if(np.isfinite(site_lats[site][row, column]) and (site_elev[site][row, column] > 5)):# and (image_data[row, column] > thresh)):
                        
                        b_lats.append(site_lats[site][row, column])
                        b_lons.append(site_lons[site][row, column])
                        b_mag.append(image_data[row, column])

        #==================================================================================================================================
        #get indicies of latitudes and longitudes
        for k in range(len(b_mag)):
            lat_inds = np.asarray((lat_ll < b_lats[k]) & (lat_ul > b_lats[k])).nonzero()
            lon_ind = np.asarray((lon_ll[lat_inds] < b_lons[k]) & (lon_lr[lat_inds] > b_lons[k])).nonzero()

            try:
                j_indx = lat_inds[0][lon_ind[0][0]]
                bright_grid[j_indx] += b_mag[k]
                bright_count[j_indx] += 1
            except:
                continue

        nz_bright = bright_count > 0
        bright_grid[nz_bright] /= bright_count[nz_bright]

        bright_grids.update({intervals:bright_grid})

        image_time = image_time + dt.timedelta(minutes = 2)
        intervals += 1

    file_time = file_time + dt.timedelta(hours = 1)

#print('brightness intervals', intervals)

#==========================================================================================================================================
#==========================================================================================================================================
#run covariance in 10 minute intervals
print('\nCovariance')

#==========================================================================================================================================
#==========================================================================================================================================
#define constant plot parameters
fg = open('grid.dat','r')
min_lat = float(fg.readline().split()[1])
max_lat = float(fg.readline().split()[1])
min_lon = float(fg.readline().split()[1])
max_lon = float(fg.readline().split()[1])

#==========================================================================================================================================
#==========================================================================================================================================
#create a cirlce overlay for the plot
theta = np.linspace(0, 2*np.pi, 360)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T

circle = mpath.Path(verts * radius + center)

#==========================================================================================================================================
#==========================================================================================================================================
#find the average of each velocity and brightness value for the full grid across all keys in the grid dictoinary
vel_stacked = np.array(list(vel_grids.values())) #place all the velocity values into one array

bright_stacked = np.array(list(bright_grids.values())) #place all the brightness values into one array
bright_stacked = np.where(bright_stacked == 0, np.nan, bright_stacked) #replace all zeros with nans

vel_avg = np.mean(vel_stacked, axis = 0) #average the velocity values together across all keys
bright_avg = np.nanmean(bright_stacked, axis = 0) #average the brightness values together across all keys

#print(vel_avg)
#print(vel_avg.shape)

#print(bright_avg)
#print(bright_avg.shape)

#==========================================================================================================================================
#==========================================================================================================================================
#loop through dictionaries and calculate covariance for each time
cov_max = 0
intervals = intervals + 5
for interval in range(intervals):
    #print(interval)
    try:
        print(rtimes[interval])

    except:
        continue

    covariance_grid = np.zeros(len(grid['index'])) #empty array to hold covariance values where radar data was considered for a given time
    #covariance_grid_nd = np.zeros(len(grid['index'])) #empty array to hold covariance values where radar data was not considered for a given time

    cov_vel_sets = np.zeros((5, 3730)) #empty array to hold the values in the five two minute velocity intervals
    cov_bright_sets = np.zeros((5, 3730)) #empty array to hold the values in the five two minute brightness intervals
    limit = interval + 5

    if limit <= intervals:
        index = 0 #initial index that the full velocity and brightness grids will be placed into

        for key in range(interval, limit):
            try:
                cov_vel_set = vel_grids[key] #full velocity grid for current set
                cov_vel_sets[index, :] = cov_vel_set #add current velocity grid to list of grids

                cov_bright_set = bright_grids[key] #full brightness grid for current set
                cov_bright_sets[index, :] = cov_bright_set #add current brightness grid to list of grids

            except:
                continue

            index += 1 #jump to the next index for the full velocoty and brightness grids to be placed into

        for j in range(grid_points):
            vel_vals = cov_vel_sets[:, j]
            bright_vals = cov_bright_sets[:, j]

            no_bright = np.all(bright_vals == 0) #checks if there are brightness values in the current interval
            is_vel = np.all(vel_vals != 0) #checks if there velocity values in the current interval
            
            #curr_vel_grid = vel_grids[interval] #velocity grid for the current time
            #curr_bright_grid = bright_grids[interval] #brightness grid for the curret time

            #curr_bright_grid = np.where(curr_bright_grid == 0, np.nan, curr_bright_grid) #replace all zeros with nans

            #calculate covariance if there are brightness values in the current interval
            if not no_bright and is_vel:
                #covariance_grid[j] = 1/len(vel_vals) * np.sum( (vel_vals - np.mean(curr_vel_grid)) * (bright_vals - np.nanmean(curr_bright_grid)) )
                covariance_grid[j] = 1/len(vel_vals) * np.sum( (vel_vals - vel_avg[j]) * (bright_vals - bright_avg[j]) )

            #skip covariance calculation if there are no brightness values in the current interval
            else:
                continue
    
    #print(vel_avg)
    #print(bright_avg)
    #print(np.max(covariance_grid))
    #print(np.min(covariance_grid))
    #print(intervals)

    if np.all(covariance_grid == 0):
        break

    #======================================================================================================================================
    #set a longitude aligned with noon and a latitude for pole
    noonlon = aacgm.inv_mlt_convert(rtimes[interval].year, rtimes[interval].month, rtimes[interval].day, rtimes[interval].hour, int(rtimes[interval].minute) - 2, 0, 12)-180
    pole_lat = 90

    #======================================================================================================================================
    #create figure
    fig = plt.figure(figsize = (10, 10), dpi = 200)

    #======================================================================================================================================
    #define coordinate reference system
    crs=ccrs.AzimuthalEquidistant(central_longitude=noonlon,
                                  central_latitude=pole_lat,
                                  false_easting=0.0,
                                  false_northing=0.0,
                                  globe=None)

    ax = fig.add_subplot(1, 1, 1, projection=crs)
    fig.set_size_inches(10, 10)

    ax.set_adjustable('datalim')
    ax.set_extent((min_lon, max_lon, min_lat, max_lat), ccrs.PlateCarree()) #setting limits of projection to be northern hemisphere
    ax.set_boundary(circle, transform=ax.transAxes) #set the boundary to be a circle instead of a square
    ax.gridlines() #add latitude and longitude grid lines
    ax.set_clip_on(True)

    lab_lat = min_lat + 2
    lon_delta = -2

    #======================================================================================================================================
    #annotate latitude markers, MLT locations, and time on plot
    transform = ccrs.PlateCarree()._as_mpl_transform(ax)
    col='k'
    lbfsize='large'
    
    if min_lat <= 60:
        ax.annotate("60", [0, 60], xycoords=transform, xytext=(0, 60), c=col, fontsize=lbfsize, zorder=10) #add 60 degree latitude line to plot
    if min_lat <= 70:
        ax.annotate("70", [0, 70], xycoords=transform, xytext=(0, 70), c=col, fontsize=lbfsize, zorder=10) #add 70 degree latitude line to plot
    if min_lat <= 80:
        ax.annotate("80", [0, 80], xycoords=transform, xytext=(0, 80), c=col, fontsize=lbfsize, zorder=10) #add 80 degree latitude line to plot
        
    ax.annotate("Noon", [.5, .85], xycoords='axes fraction', xytext=(.5, .97), horizontalalignment='center', c=col, fontsize=lbfsize, zorder=10)
    ax.annotate("Dusk", [.5, .85], xycoords='axes fraction', xytext=(.05, .5), horizontalalignment='center', c=col, fontsize=lbfsize, zorder=10)
    ax.annotate("Midnight", [.5, .85], xycoords='axes fraction', xytext=(.5, .03), horizontalalignment='center', c=col, fontsize=lbfsize, zorder=10)
    ax.annotate("Dawn", [.5, .85], xycoords='axes fraction', xytext=(.95, .5), horizontalalignment='center', c=col, fontsize=lbfsize, zorder=10)

    xpos = .05
    ypos = .97
        
    plot_time = rtimes[interval].strftime('%Y%m%d-%H:%M:%S') #format the time string to be placed on the plot
    ax.annotate(plot_time, (xpos, ypos), xycoords = 'axes fraction')

    #======================================================================================================================================
    #call colormap, normalization function, and create patches
    colormap = mpl.colormaps['RdYlGn'] #get colormap object from matplotlib colormap
    grey = (100/255, 100/255, 100/255) #define a grey color to plot where there is no covariance value
    norm = Normalize(vmax = 350000, vmin = -350000) #normalization function for images

    patches = [] #patches where radar data was used

    #======================================================================================================================================
    #normalize covariance where no radar data was considered and place it into patches
    for l in range(len(grid['index'])):
        x = lon_ll[l]
        y = lat_ll[l]
        width = lon_lr[l] - lon_ll[l]
        height = lat_ul[l] - lat_ll[l]

        #color = colormap(norm(covariance_grid_nd[l])) #add a color to a normalized covariance value
        #patches.append(Rectangle((x, y), width, height, color = color, ec = None, alpha = .75, zorder = 2)) #add covariance data to patches to be plotted on map; ec and edgecolor are interchangeable
        
    #p = PatchCollection(patches, transform = ccrs.PlateCarree(), match_original=True)
    #ax.add_collection(p)

    #======================================================================================================================================
    #normalize covariance where radar data was considered and place it into patches
    for l in range(len(grid['index'])):
        x = lon_ll[l]
        y = lat_ll[l]
        width = lon_lr[l] - lon_ll[l]
        height = lat_ul[l] - lat_ll[l]

        if covariance_grid[l] == 0:
            patches.append(Rectangle((x, y), width, height, color = grey, ec = None)) #color the box gray if there is no data at the given point

        else:
            color = colormap(norm(covariance_grid[l])) #add a color to a normalized covariance value
            patches.append(Rectangle((x, y), width, height, color = color, ec = None, zorder = 2.5)) #add covariance data to patches to be plotted on map; ec and edgecolor are interchangeable
        
    p = PatchCollection(patches, transform = ccrs.PlateCarree(), match_original=True)
    ax.add_collection(p)

    #======================================================================================================================================
    #add and resize colorbar
    cbar_ax = fig.add_axes([0, 0, .1, .1]) #left, bottom, width, and height of axis; maximum of 1 for each value
    cbar_ax.set_position([0.9, .01, .02, .3]) #left, bottom, width, and height; set position of colorbar

    cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), cax = cbar_ax, shrink = 0.5)
    cbar.ax.set_ylabel('Covariance (Unitless)')
    ticks = cbar.get_ticks()
    cbar.ax.set_yticklabels([f'{tick:.0e}' for tick in ticks])
    #cbar.ax.set_yticklabels(['-9e5', '0', '9e5'])

    #======================================================================================================================================
    #plot meridians if flag is raised
    if meridians != None:
        for meridian in meridians:

            print("plotting meridian:",meridian)
            meridlon = aacgm.inv_mlt_convert(rtimes[interval].year, rtimes[interval].month, rtimes[interval].day, int(rtimes[interval].hour), int(rtimes[interval].minute), 0, meridian)
        
            ax.plot([meridlon, meridlon], [min_lat+2, max_lat-2], transform = ccrs.PlateCarree(), color = 'k', lw = 1.5)

            label=cnv_mlt_label(meridian)
            ax.annotate(label, [meridlon + lon_delta, lab_lat], xycoords = transform, xytext = (meridlon + lon_delta, lab_lat),c = col, clip_on = True)  

    #======================================================================================================================================
    #title and save figure
    #name_time = dt.datetime(image_time.year, image_time.month, image_time.day, image_time.hour, image_time.minute, image_time.second)
    DMY_str = rtimes[interval].strftime('%Y%m%d')#string of the day, month, and year
    HM_str = rtimes[interval].strftime("%H%M") #string of the hour and minute
    name = DMY_str + '_' + HM_str + '_covariance.png'
    plt.savefig(name)
    print('Plot has been made\n')
    
    ax.cla()
    cbar.ax.cla()
    plt.close()

#print(cov_max)
print(sttime, ndtime)
#==========================================================================================================================================
#==========================================================================================================================================