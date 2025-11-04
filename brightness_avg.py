#!/usr/bin/env python3

#THIS SCRIPT GETS AN AVERAGE AROURAL BRIGHTNESS OVER A LATITUDE RANGE FROM THE ASI IMAGERS
#TO CALL THE SCRIPT: MLT, Start Time, End Time (22 201403030500 201403031100)

#import required modules
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import scipy.io as sio
import glob
import datetime as dt

from matplotlib.collections import PolyCollection
from matplotlib import colors as mpl_colors

import cdflib
from cdflib import cdfread
from cdflib.xarray import cdf_to_xarray

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.dates as mdates

from map_utils import min_sep


from pwr_col_cmap import pwr_col
cmap=pwr_col()

import time as tmr

import aacgm
from date_strings import make_date_str, make_date_time_str,cnv_datetimestr_dtlist,cnv_datetimestr_datetime


#get data
def cdf_read_file(cdf_f,site,start_time,end_time):
    var='thg_asf_'+site
    sttime=[start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second, 0]
    ndtime=[end_time.year, end_time.month, end_time.day, end_time.hour, end_time.minute, end_time.second+delta_t, 0]
    img=cdf_f.varget(var,starttime=sttime,endtime=ndtime)
    if (not isinstance(img, (list, tuple, np.ndarray))) or (img is None):
        return(-1)

    return(img)


#get fov data
def thg_asi_fov(site,time,mag=False,alt=1):
    fov_path='/import/SUPERDARN/themis_asi/fovMapping/' #path to fov data
    fname=glob.glob(fov_path+'thg_l2_asc_'+site+'*') #name of fov file
    print(fname[0])

    cal_f=cdfread.CDF(fname[0])

    sttime=[time.year, time.month, time.day, time.hour, time.minute, time.second, 0] #start time
    print('mag = ',mag)
    print(sttime)
    if mag:
        sta_lat=cal_f.varget('thg_asc_'+site+'_mlat',starttime=sttime) #station latitude
        sta_lon=cal_f.varget('thg_asc_'+site+'_mlon',starttime=sttime) #station longitude
        print(sta_lat, sta_lon)
        if (sta_lat is None):
            return(-1)

        lats=cal_f.varget('thg_asf_'+site+'_mlat',starttime=sttime)[0] #latitudes in asi data
        lons=cal_f.varget('thg_asf_'+site+'_mlon',starttime=sttime)[0] #longitudes in asi data
    else:
        sta_lat=cal_f.varget('thg_asc_'+site+'_glat',starttime=sttime) #station latitude
        sta_lon=cal_f.varget('thg_asc_'+site+'_glon',starttime=sttime) #station longitude
        print(sta_lat, sta_lon)
        if (sta_lat is None):
            return(-1)

        try:
            lats=cal_f.varget('thg_asf_'+site+'_glat',starttime=sttime)[0] #latitudes in asi data
        except:
            return(-1)
        try:
            lons=cal_f.varget('thg_asf_'+site+'_glon',starttime=sttime)[0] #longitudes in asi data
        except:
            return(-1)

    lats=np.delete(lats,256,0)
    lats=np.delete(lats,256,1)
    lons=np.delete(lons,256,0)
    lons=np.delete(lons,256,1)

    elev=cal_f.varget('thg_asf_'+site+'_elev',starttime=sttime)[0] #station elevation
    return(sta_lat,sta_lon,lats,lons,elev)


#set arguments for use in termial command line
parser = argparse.ArgumentParser(description='sector_gdf_vel argument parser')
parser.add_argument('plot_mlt', type=float) #argument of MLT; write as integer betweem 00 and 23
parser.add_argument("sttime", type=str) #start time argument; YYYYmmddHHMM
parser.add_argument("ndtime", type=str) #end time argument; YYYYmmddHHMM
parser.add_argument('--lat_min', required=False, type=float, default=60) #minimum latitude, optional
parser.add_argument('--lat_max', required=False, type=float, default=80) #maximum latitude, optional


args = parser.parse_args()
plot_mlt=args.plot_mlt
lat_min=args.lat_min
lat_max=args.lat_max
sttime=args.sttime
ndtime=args.ndtime

mag = True #if True, script is using magnetic time


#define start and end times
start_time=cnv_datetimestr_datetime(sttime)
end_time=cnv_datetimestr_datetime(ndtime)


#define change in latitude and longitude
dlat=.15
dlon=1*dlat/np.cos((90-(lat_max-lat_min)/2)*np.pi/180.)

delta_t=3

nlat=int((lat_max-lat_min)/dlat) #number of latitudes
n_intervals=int(((end_time-start_time).total_seconds())/delta_t) #number of intervals

plats = np.zeros((nlat, n_intervals))
ptimes=np.zeros((nlat,n_intervals))

for j in range(n_intervals):
    plats[:, j] = lat_min + np.arange(nlat) * dlat


#list station sites
#sites=['kian','atha','fykn','inuv','whit','gako','fsmi','fsim','gill','rank','kuuj','snkq'] #every site
sites = ['whit'] #single site
scales={'atha':2000,'fsim':4500,'fykn':2000,'inuv':2000,'whit':2500,'fsmi':5000,'gako':2000,'gill':2500,'rank':4000,'kian':3000, 'snkq':4000}

n_hours=int(np.rint((end_time-start_time).seconds/3600)) #number of hours


#get asi data
datestr=start_time.strftime("%Y%m%d")
#data_path=os.environ["HOME"]+'/import/SUPERDARN/themis_asi/asi_cdfs/'+datestr+'/'
data_path='/import/SUPERDARN/themis_asi/asi_cdfs/'+datestr+'/'

time=start_time
for jt in range(n_intervals):
    time=time+dt.timedelta(seconds=delta_t)
    ptimes[:,jt]=mdates.date2num(time)


#loop through sites and print data
img_avgs = [] #empty array to hold brightness averages over entire time series
saved_times = [] #empty array to hold times where there is a brightness
for site in sites:
    #print(site)
    time=start_time
    file_time=start_time
    jout=0
    for jh in range(n_hours):

        datestr=file_time.strftime("%Y%m%d%H")
        cdf_name=data_path+'thg_l1_asf_'+site+'_'+datestr+'_v01.cdf'
        print(jh,file_time,cdf_name)
        file_time=file_time+dt.timedelta(hours=1)

        if not os.path.exists(cdf_name):
            print("FILE NOT FOUND: ",cdf_name)
            continue
        
        cdf_f=cdfread.CDF(cdf_name)
        imgs = cdf_read_file(cdf_f,site,start_time,end_time)
        
        #print('the imgs variable returns', imgs)

        try:
            data=cdf_to_xarray(cdf_name,to_datetime=True)
        except:
            print("NO DATA: ",cdf_name)
            continue

        imgs_name='thg_asf_'+site
        times_name=imgs_name+'_epoch'
        #imgs=data[imgs_name].values
        ns=1.e-9
        times=data[times_name].values
        
        n_img=len(imgs)
        
        try:
            sta_lat,sta_lon,lats,lons,elev=thg_asi_fov(site,time,mag=mag)
        except:
            print('problem with fov')
            continue
       
        for j in range(n_img):
            image_data=imgs[j]

            time=dt.datetime.utcfromtimestamp(times[j].astype(int)*ns)
            
            plon=aacgm.inv_mlt_convert(time.year,time.month,time.day,int(time.hour),int(time.minute),int(time.second),plot_mlt)
            if plon < 0: plon+=360
            print(time,plon)
            print(site)

            finelons=np.isfinite(lons)
            neglons=lons<0
            lons[neglons]+=360

            pixls=(lons > plon-dlon) & (lons < plon+dlon) & (elev > 5)
            px_lats=lats[pixls] #latitudes of the pixels in the image

            img_pixls = image_data[pixls] - 2600 #brightness value of aurora
            plat = 70 #latitude for which the range is centered around
            dlat = .5 #change in latitude used to define range
            pixls2 = (px_lats > plat-dlat) & (px_lats < plat + dlat) #pixels within the latitude range

            #print(px_lats) #prints the latitudes of the pixels in the image
            #print(img_pixls) #prints the array of brightness values
            #print(px_lats[pixls2], img_pixls[pixls2]) #prints the brigtness values in the latitude range

            try:
                img_avg = sum(img_pixls[pixls2]) / len(img_pixls[pixls2]) #averages the brightness values
                print('Brightness Average:', img_avg) #prints a single brightness value representing the average over the latitude range
                img_avgs.append(img_avg) #adds the brightness average for a single time to an array to hold all averages over the time series
                saved_times.append(time) #adds the current times to an array to be plotted
            except:
                print('Division by Zero')
                continue

print(np.round(img_avgs, 2))
print(len(img_avgs))
print(len(saved_times))
xdm = 20
ydm = 9
fsz = 'large'

fig = plt.figure(figsize = (xdm, ydm))
ax = fig.add_subplot()

ax.plot(saved_times, img_avgs)

locator = mdates.AutoDateLocator(minticks = 3, maxticks = 12)
formatter = mdates.ConciseDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.set_xlabel('Time (UT)')
ax.set_ylabel('Auroral Brightness')
ax.set_title(str(plot_mlt)+' MLT '+sttime+'-'+ndtime+' Brightness at '+site)

pname = sttime+'-'+ndtime+'_themis_brightness.png'
pname = 'brightness_plot.png'

#ax.plot(img_avgs)
plt.savefig(pname)
print('Plot has been made')
