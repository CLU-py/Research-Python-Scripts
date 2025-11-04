#!/usr/bin/env python3

#THIS SCRIPT GETS THE BRIGHTNESS AT ALL MERIDIANS WITH A ONE DEGREE LATITUDE BIN AND PLOTS IT ON A HEMISPHERICAL MAP
#TO CALL THIS SCRIPT: starttime endtime (201403210400 201403210600)
#IF SCRIPT IS NOT EXECUTABLE IN TERNIMAL, RUN chmod +x {filename}.py

#TO RUN PROCESS IN SCREEN MODE: 'screen' TO START SCREEN SESSION, crtl + a (release), then d TO DETACH THE PROCESS/SCREEN
#TO RESUME DETACHED PROCESS: 'screen -r' IN COMMAND LINE

#TO SEE PROCESSES RUNNING IN TERMINAL: 'top' IN COMMAND LINE

#MUST HAVE ASI CDFS DOWNLOADED TO ASI CDF DIRECTIORY: /import/SUPERDARN/themis_asi/asi_cdfs/{event date}
#TO DOWNLOAD CDFS: write_wget_list.py {sttime} {ndtime} > list, wget -i list

#==========================================================================================================================================
#==========================================================================================================================================
#import required modules
import numpy as np
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import argparse
import aacgm
import glob
import os
import datetime as dt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from map_grid import make_grid #refer to map_grid.py
from datetime import timedelta
from datetime import datetime

import cartopy .crs as ccrs
import cartopy.feature as cfeature

import cdflib
from cdflib import cdfread
from cdflib.xarray import cdf_to_xarray
from date_strings import make_date_str, make_date_time_str, cnv_datetimestr_dtlist, cnv_datetimestr_datetime

#==========================================================================================================================================
#==========================================================================================================================================
#add commands to call in terminal when calling script; seae top of script for command order
parser = argparseparser = argparse.ArgumentParser(description='sector_gdf_vel argument parser')
parser.add_argument('sttime', type = str) #start time argument; YYYmmddHHMM
parser.add_argument('ndtime', type = str) #end time argument; YYYmmddHHMM
parser.add_argument('--noAverage', required = False, dest = 'noAverage', action = 'store_false', default = True) #tells the script to not average the brightness images
parser.add_argument('--meridians', type = float, nargs = "+", default = None)

args = parser.parse_args()
sttime = args.sttime
ndtime = args.ndtime
avg = args.noAverage
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
#define function to read cdf file
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
#define function to get site fov
def thg_asi_fov(site, time, mag = False, alt = 1):
    fov_path = '/import/SUPERDARN/themis_asi/fovMapping/' #path to fov data on SuperDARN-proc
    #fov_path = '/tank/storage/SUPERDARN/data/themis_asi/fovMapping/' #path to fov data on SuperDARN-2
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

mag = True #if True, script is using magnetic local time (MLT)

#==========================================================================================================================================
#==========================================================================================================================================
#define start and end times
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
#sites = ['kian', 'atha', 'fykn', 'inuv', 'gako', 'fsmi', 'fsim', 'gill', 'rank', 'kuuj', 'snkq'] #every asi site
scales = {'atha':2000, 'fsim':4500, 'fykn':2000, 'inuv':2000, 'whit':2500, 'fsmi':5000, 'gako':2000, 'gill':2500, 'rank':4000, 'kian':3000, 'snkq':4000}

n_hours = int(np.rint((end_time - start_time).seconds/3600)) #number of hours in bewtween the start time and end time

#==========================================================================================================================================
#==========================================================================================================================================
#get asi data
datestr = start_time.strftime("%Y%m%d")
data_path = '/import/SUPERDARN/themis_asi/asi_cdfs/'+datestr+'/' #data path if running on SuperDARN-proc
#data_path = '/tank/storage/SUPERDARN/data/themis_asi/asi_cdfs/'+datestr+'/' #data path if running on SuperDARN-2

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
#setup grid
grid = make_grid()

lat_ll = grid['lat_ll'] #list of lower left latitudes
lat_ul = grid['lat_ul'] #list of upper left latitudes

lon_ll = grid['lon_ll'] #list of lower left longitudes
lon_lr = grid['lon_lr'] #list of lower right longitudes

#==========================================================================================================================================
#==========================================================================================================================================
#define constant plot parameters
#set minimum and maximum latitude and longitude values
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
    #average brightness data into two minute intervals
    #if times is within current image time and image time plus 2, add it into the average
    #======================================================================================================================================
    #loop through two minute intervals
    for i in range(minute_intervals):
        print(image_time)

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
            #normalize data from each site
            nrows = site_lats[site].shape[0] #number of rows; 256
            ncols = site_lats[site].shape[1] #number of columns; 256

            dark_image = np.zeros((nrows, ncols))
            try:
                for jr in range(nrows):
                    for jc in range(ncols):
                        dark_image[jr, jc] = np.nanmin(imgs[site][:][jr, jc])

                if avg:
                    image_data = img_avg - dark_image

                else:
                    image_data = imgs[site][i*40] - dark_image

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

        #==================================================================================================================================
        #set a longitude aligned with noon and a latitude for pole
        noonlon = aacgm.inv_mlt_convert(image_time.year, image_time.month, image_time.day, image_time.hour, int(image_time.minute) - 2, 0, 12)-180
        pole_lat = 90

        #==================================================================================================================================
        #create figure
        fig = plt.figure(figsize = (10, 10), dpi = 200)

        #==================================================================================================================================
        #define coordinate reference system
        crs=ccrs.AzimuthalEquidistant(central_longitude=noonlon,
                                  central_latitude=pole_lat,
                                  false_easting = 0.0,
                                  false_northing = 0.0,
                                  globe = None)

        ax = fig.add_subplot(1, 1, 1, projection = crs)
        fig.set_size_inches(10, 10)

        ax.set_adjustable('datalim')
        ax.set_extent((min_lon, max_lon, min_lat, max_lat), ccrs.PlateCarree()) #setting limits of projection to be northern hemisphere
        ax.set_boundary(circle, transform = ax.transAxes) #set the boundary to be a circle instead of a square
        ax.gridlines() #add latitude and longitude grid lines
        ax.set_clip_on(True)

        #==================================================================================================================================
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

        xpos=.05
        ypos=.97
        
        title_time = image_time.strftime('%Y%m%d-%H:%M:%S') #format the time string to be placed on the plot
        ax.annotate(title_time, (xpos,ypos), xycoords='axes fraction')

        lab_lat = min_lat+2
        lon_delta = -2

        #==================================================================================================================================
        #call colormap, normalization function, and create patches
        colormap = mpl.colormaps['rainbow'] #get colormap object from matplotlib colormap
        grey = (100/255, 100/255, 100/255) #define a grey color to plot where there is no value from the ASI imagers
        norm = Normalize(vmax = 5000, vmin = 0) #normalization function for images

        patches = [] #patches where radar data was used

        nz_bright = bright_count > 0
        bright_grid[nz_bright] /= bright_count[nz_bright]
        
        #==================================================================================================================================
        #normalize brightness data and place it into patches
        #bright_grid = np.where(bright_grid == 0, np.nan, bright_grid)
        for l in range(len(grid['index'])):
            x = lon_ll[l]
            y = lat_ll[l]
            width = lon_lr[l] - lon_ll[l]
            height = lat_ul[l] - lat_ll[l]
        
            if bright_grid[l] == 0:
                patches.append(Rectangle((x, y), width, height, color = grey, ec = None)) #color the box gray if there is no data at the given point

            else:
                color = colormap(norm(bright_grid[l])) #add a color to a normalized brightness value
                patches.append(Rectangle((x, y), width, height, color = color, ec = None)) #add brightness data to patches to be plotted on map; ec and edgecolor are interchangeable
        
        p = PatchCollection(patches, transform = ccrs.PlateCarree(), match_original=True)
        ax.add_collection(p)

        #==================================================================================================================================
        #add and resize colorbar
        cbar_ax = fig.add_axes([0, 0, .1, .1]) #left, bottom, width, and height of axis; maximum of 1 for each value
        cbar_ax.set_position([0.9, .01, .02, .3]) #left, bottom, width, and height; set position of colorbar

        cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = colormap), cax = cbar_ax, shrink = 0.5)
        cbar.ax.set_ylabel('Auroral Brightness (Unitless)')

        #======================================================================================================================================
        #plot meridians if flag is raised
        if meridians != None:
            for meridian in meridians:

                print("plotting meridian:",meridian)
                meridlon = aacgm.inv_mlt_convert(image_time.year, image_time.month, image_time.day, int(image_time.hour), int(image_time.minute), 0, meridian)
        
                ax.plot([meridlon, meridlon], [min_lat+2, max_lat-2], transform = ccrs.PlateCarree(), color = 'k', lw = 1.5)

                label=cnv_mlt_label(meridian)
                ax.annotate(label, [meridlon + lon_delta, lab_lat], xycoords = transform, xytext = (meridlon + lon_delta, lab_lat),c = col, clip_on = True) 

        #==================================================================================================================================
        #title and save figure
        #name_time = dt.datetime(image_time.year, image_time.month, image_time.day, image_time.hour, image_time.minute, image_time.second)
        DMY_str = image_time.strftime('%Y%m%d')#string of the day, month, and year
        HM_str = image_time.strftime("%H%M") #string of the hour and minute
        
        if avg:
            name = DMY_str + '_' + HM_str + '_brightness.png'
        
        else:
            name = 'noAverage_' + DMY_str + '_' + HM_str + '_brightness.png'
        
        plt.savefig(name)
        print('Plot has been made\n')
        plt.close()
    
        ax.cla()
        image_time = image_time + dt.timedelta(minutes = 2)

    file_time = file_time + dt.timedelta(hours = 1)

#==========================================================================================================================================
#==========================================================================================================================================