#!/usr/bin/env python3

#THIS SCRIPT GETS BOTH BRIGHTNESS AND VELOCITY DATA OVER THE FULL LATITUDE RANGE AND RUNS COVARIANCE OR CORRELATION BASED ON THE SPECIFIED FLAGS
#TO CALL THIS SCRIPT: MLT fileName starttime endtime (22 vel_out 201403210400 201403210600 --cov (--corr))
#IF SCRIPT IS NOT EXECUTABLE IN TERNIMAL, RUN chmod +x {filename}.py

#==========================================================================================================================================
#==========================================================================================================================================
#import required modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
import glob
import datetime

from matplotlib.colors import Normalize

from cdflib import cdfread
from cdflib.xarray import cdf_to_xarray

from df_vel_class import df_vel
from date_strings import make_date_str, make_date_time_str, cnv_datetimestr_dtlist, cnv_datetimestr_datetime
import aacgm
import read_df_record

from pwr_col_cmap import pwr_col
cmap = pwr_col()

import faulthandler
faulthandler.enable()
#==========================================================================================================================================
#==========================================================================================================================================
#add commands to call in terminal when calling script; seae top of script for command order
parser = argparseparser = argparse.ArgumentParser(description='sector_gdf_vel argument parser')
parser.add_argument('plot_mlt', type=float) #argument of MLT; write as integer between 00 and 23
parser.add_argument('fname', type=str) #file name of vel_out file
parser.add_argument('sttime', type=str) #start time argument; YYYmmddHHMM
parser.add_argument('ndtime', type=str) #end time argument; YYYmmddHHMM

parser.add_argument('--corr', required=False, dest='corr', action='store_true', default=False) #argument telling script to run correlation between brightness and velocity instead of covariance
parser.add_argument('--lines', required=False, dest='lines', action='store_true', default=False) #argument telling script to plot lines across the figures

parser.add_argument('--mxval', required=False, type=int, default=20000)
parser.add_argument('--global', required=False, dest='global_v', action='store_true', default=False)
parser.add_argument('--min_lat', required=False, dest='in_min_lat', type=float, default=None)
parser.add_argument('--max_lat', required=False, dest='in_max_lat', type=float, default=None)
parser.add_argument('--lat_min', required=False, type=float, default=60)
parser.add_argument('--lat_max', required=False, type=float, default=80)

args = parser.parse_args()
fname = args.fname
plot_mlt = args.plot_mlt
mlt = plot_mlt
sttime = args.sttime
ndtime = args.ndtime

corr = args.corr
fig_lines = args.lines

mxval = args.mxval
global_v = args.global_v
in_min_lat = args.in_min_lat
in_max_lat = args.in_max_lat
lat_min = args.lat_min
lat_max = args.lat_max

#==========================================================================================================================================
#==========================================================================================================================================
#import extra packages and set parameters
import datetime as dt
from datetime import datetime

np.set_printoptions(threshold = 1000) #this is the default value for threshols
#np.set_printoptions(threshold = np.inf)

#==========================================================================================================================================
#==========================================================================================================================================
#define function to get fov and brightness data
def cdf_read_file(cdf_f,site,start_time,end_time):
    var='thg_asf_'+site
    sttime=[start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second, 0]
    ndtime=[end_time.year, end_time.month, end_time.day, end_time.hour, end_time.minute, end_time.second+delta_t, 0]
    img=cdf_f.varget(var,starttime=sttime,endtime=ndtime)
    if (not isinstance(img, (list, tuple, np.ndarray))) or (img is None):
        return(-1)

    return(img)

def thg_asi_fov(site,time,mag=False,alt=1):
    fov_path='/import/SUPERDARN/themis_asi/fovMapping/' #path to fov data
    fname=glob.glob(fov_path+'thg_l2_asc_'+site+'*') #name of fov file
    print(fname[0])

    cal_f=cdfread.CDF(fname[0])

    sttime=[time.year, time.month, time.day, time.hour, time.minute, time.second, 0] #start time
    #print('mag = ',mag)
    #print('Start time:', sttime)
    if mag:
        sta_lat=cal_f.varget('thg_asc_'+site+'_mlat',starttime=sttime) #station latitude
        sta_lon=cal_f.varget('thg_asc_'+site+'_mlon',starttime=sttime) #station longitude
        #print('Station location:', sta_lat, sta_lon)
        if (sta_lat is None):
            return(-1)

        lats=cal_f.varget('thg_asf_'+site+'_mlat',starttime=sttime)[0] #latitudes in asi data
        lons=cal_f.varget('thg_asf_'+site+'_mlon',starttime=sttime)[0] #longitudes in asi data
    else:
        sta_lat=cal_f.varget('thg_asc_'+site+'_glat',starttime=sttime) #station latitude
        sta_lon=cal_f.varget('thg_asc_'+site+'_glon',starttime=sttime) #station longitude
        #print('Station location:', sta_lat, sta_lon)
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

mag = True #if True, script is using magnetic local time (MLT)

#==========================================================================================================================================
#==========================================================================================================================================
#define start and end times
start_time=cnv_datetimestr_datetime(sttime)
end_time=cnv_datetimestr_datetime(ndtime)

#==========================================================================================================================================
#==========================================================================================================================================
#define change in latitude and longitude
dlat=.15
dlon=1*dlat/np.cos((90-(lat_max-lat_min)/2)*np.pi/180.)

delta_t=3

nlat=int((lat_max-lat_min)/dlat) #number of latitudes
n_intervals=int(((end_time-start_time).total_seconds())/delta_t) #number of intervals

ptimes=np.zeros((nlat,n_intervals))

#==========================================================================================================================================
#==========================================================================================================================================
#list station sites
#sites=['atha'] #single asi site
sites=['kian','atha','fykn','inuv','whit','gako','fsmi','fsim','gill','rank','kuuj','snkq'] #every asi site
scales={'atha':2000,'fsim':4500,'fykn':2000,'inuv':2000,'whit':2500,'fsmi':5000,'gako':2000,'gill':2500,'rank':4000,'kian':3000, 'snkq':4000}

n_hours=int(np.rint((end_time-start_time).seconds/3600)) #number of hours

#==========================================================================================================================================
#==========================================================================================================================================
#get asi data
datestr=start_time.strftime("%Y%m%d")
data_path='/import/SUPERDARN/themis_asi/asi_cdfs/'+datestr+'/'

time=start_time
for jt in range(n_intervals):
    time=time+dt.timedelta(seconds=delta_t)
    ptimes[:,jt]=mdates.date2num(time)

#==========================================================================================================================================
#==========================================================================================================================================
#make array for latitudes with a dlat of one degree
latitudes = np.arange(60, 81, 1) #create an array of latitudes with a step of 1

plot_img2 = np.zeros((len(latitudes),n_intervals))
counts2 = np.zeros((len(latitudes),n_intervals))

min_lat = np.min(latitudes) #minimum latitude
max_lat = np.max(latitudes) #maximum latitude

#loop through sites and print data
for site in sites:
    print('\n', site)
    brightness_times = []

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
        #print(times)
        #print(type(times))

        n_img=len(imgs)

        try:
            sta_lat,sta_lon,lats,lons,elev=thg_asi_fov(site,time,mag=mag)
        except:
            print('problem with fov')
            continue

        for j in range(n_img):
            image_data=imgs[j]

            time=dt.datetime.utcfromtimestamp(times[j].astype(int)*ns)
            #time = dt.datetime.fromtimestamp(times[j].astype(int)*ns, dt.UTC)

            plon=aacgm.inv_mlt_convert(time.year,time.month,time.day,int(time.hour),int(time.minute),int(time.second),mlt)
            if plon < 0: plon+=360
            #print(time,plon)

            finelons=np.isfinite(lons)
            neglons=lons<0
            lons[neglons]+=360

            pixls=(lons > plon-dlon) & (lons < plon+dlon) & (elev > 5)
            px_lats = lats[pixls] #planetary latitudes of the pixels in the image
            brightness_times.append(time) #add current time to time array

            if len(px_lats) == 0:
                time=time+dt.timedelta(seconds=delta_t)
                continue

            n_pix=len(px_lats) #number of pixels where there is brightness data

            img_pixls=image_data[pixls] #brightness values where there is brightness along the specified meridian
            #print(img_pixls)
            
            img02=np.zeros(len(latitudes))
            counts02=np.zeros(len(latitudes)) #counts number of latitude points; 55-88, dlat = .15

            latbin = np.zeros(len(px_lats))

            ngood=0

            for i, val in enumerate(px_lats):
                if val < min_lat or val > max_lat:
                    continue

                latbin[i] = (latitudes[np.abs(latitudes-val).argmin()]) #find the closest latitude point to each brightness latitude

                index = np.where(latitudes == latbin[i])[0]

                counts02[index] += 1
                img02[index] += img_pixls[i]

                ngood += 1

            a = counts02 > 0
            img02[a] /= counts02[a]

            #normalize the brightness values to be consistent over all imagers
            min_img = np.min(img02[a])
            img02[a] = (img02[a] - min_img)/min_img

            plot_img2[:,jout] += img02

            counts02[a] /= counts02[a]
            counts2[:,jout] += counts02

            if ngood == 0: continue

            tms = ptimes[0,:] < mdates.date2num(time)
            jout = len(ptimes[0,tms])

            jout += 1

            time = time+dt.timedelta(seconds=delta_t)

#==========================================================================================================================================
#==========================================================================================================================================
c_scale=5

a = counts2 > 0
plot_img2[a] /= counts2[a]

min_nz = np.min(plot_img2[a])
plot_img2[a] -= min_nz

mean = np.mean(plot_img2[a])
b=plot_img2 > (c_scale*mean)

plot_img2[b] = c_scale*mean

if np.max(plot_img2) != 0:
    plot_img2 /= np.max(plot_img2)

#==========================================================================================================================================
#==========================================================================================================================================
asi_scale=np.max(plot_img2)# /2
# asi_scale=0.1
norm=Normalize(vmax=asi_scale,vmin=0)

print('\nVelocity')
#stop
#==========================================================================================================================================
#==========================================================================================================================================
import datetime
from datetime import timedelta
#get velocity data
dfVel=df_vel(fname,global_vel=global_v)

if mag:
    aacgm.set_datetime(dfVel.dstr.yr,dfVel.dstr.mo,dfVel.dstr.dy,int(dfVel.dstr.hr),int(dfVel.dstr.mt),0)

    for j in range(dfVel.dstr.num):
        pos=aacgm.convert(dfVel.dstr.vec_lat[j],dfVel.dstr.vec_lon[j],300,0)
        dfVel.dstr.vec_lat[j]=pos[0]
        dfVel.dstr.vec_lon[j]=pos[1]

dtor=np.pi/180
inds=[]
dlon=.75 #maybe have to increase this
for idx,lon in enumerate(dfVel.dstr.vec_lon):
    lat=dfVel.dstr.vec_lat[idx]
    p_lon = aacgm.inv_mlt_convert(dfVel.dstr.yr,dfVel.dstr.mo,dfVel.dstr.dy,int(dfVel.dstr.hr),int(dfVel.dstr.mt),0,mlt)

    if p_lon<0: p_lon+=360

    if (lon > p_lon-dlon/(2*np.cos(lat*dtor))) and (lon < p_lon+dlon/(2*np.cos(lat*dtor))):
        inds.append(idx)

lats=dfVel.dstr.vec_lat[inds]

if in_min_lat != None:
    min_lat=in_min_lat
else:
    min_lat=np.min(lats)

if in_max_lat != None:
    max_lat=in_max_lat
else:
    max_lat=np.max(lats)

nlats = int((max_lat-min_lat)/dlat)
vel_x = np.zeros(nlats)
vel_y = np.zeros(nlats)
count_ar = np.zeros(nlats)
lat_ar = np.linspace(min_lat, max_lat, nlats)

rtimes = []

if isinstance(sttime, list):
    sttime = sttime[0]

if isinstance(ndtime, list):
    ndtime = ndtime[0]

st_tm = cnv_datetimestr_dtlist(sttime)
start_datetime = datetime.datetime(st_tm[0], st_tm[1], st_tm[2], st_tm[3], st_tm[4])

nd_tm = cnv_datetimestr_dtlist(ndtime)
end_datetime = datetime.datetime(nd_tm[0], nd_tm[1], nd_tm[2], nd_tm[3], nd_tm[4])

#entry = True

#==========================================================================================================================================
#==========================================================================================================================================
#calculte x and y velocities
velocity_times = [] #empty array to hold times where there are velocities

latitudes = np.arange(60, 81, 1) #create an array of latitudes with a step of 1

min_lat = np.min(latitudes) #minimum latitude
max_lat = np.max(latitudes) #maximum latitude

column = 0 #initial index of column the velocity values will be added to
#vel_intervals = #number of velocity time intervals
plot_vel = np.zeros((len(latitudes), 1000))

while 1:
    vel_x[:] = 0.0
    vel_y[:] = 0.0
    count_ar[:] = 0.0

    rtime = datetime.datetime(dfVel.dstr.yr, dfVel.dstr.mo, dfVel.dstr.dy,
                              dfVel.dstr.hr, dfVel.dstr.mt, int(dfVel.dstr.sc))

    print(rtime) #time velocity data is being taken from
    if rtime < start_datetime:
        dfVel.read_record()
        if not isinstance(dfVel.dstr, read_df_record.record):
            break

        continue

    if rtime > end_datetime:
        break

    rtimes.append(rtime)

    inds = [] #set of points along meridian
    for idx, lon in enumerate(dfVel.dstr.vec_lon):
        lat = dfVel.dstr.vec_lat[idx]
        p_lon = aacgm.inv_mlt_convert(dfVel.dstr.yr,dfVel.dstr.mo,dfVel.dstr.dy,int(dfVel.dstr.hr),int(dfVel.dstr.mt),0,mlt)
        if p_lon < 0 : p_lon += 360

        if (lon > p_lon-dlon/(2*np.cos(lat*dtor))) and (lon < p_lon + dlon/(2*np.cos(lat*dtor))):
            inds.append(idx)

    #get x and y velocity values at each time to calculate total velocity
    vx_marid = dfVel.dstr.vx[inds] #x-velocities across entire meridian (55 to 88 latitude)
    vy_marid = dfVel.dstr.vy[inds] #y-velocities across entire meridian (55 to 88 latitude)

    #calculate total velocity
    vxy_marid = np.sqrt(vx_marid**2 + vy_marid**2) #calculate the velocity magnitude across the entire meridian

    #get all latitude values for each time
    lats = dfVel.dstr.vec_lat[inds] #latitudes where there is velocity at each time

    vel0 = np.zeros(len(latitudes)) #empty meridional array for velocity values
    count0 = np.zeros(len(latitudes)) #empty counter to count how many velocity values fall in each bin

    latbin = np.zeros(len(lats)) #empty array to bin the velocity values into

    for i, val in enumerate(lats):
        if val < min_lat or val > max_lat:
            continue
        latbin[i] = (latitudes[np.abs(latitudes - val).argmin()]) #find the closest latitude point to each velocity latitude

        index = np.where(latitudes == latbin[i])[0] #find the index of the latitude array that is equal to the current lat bin

        count0[index] += 1 #count up each time a velocity value falls in the same bin
        vel0[index] += vxy_marid[i] #add the velocity value to the index for the given latitude bin

    a = count0 > 0 #find everywhere multiple velocity values fell into the same bin
    vel0[a] /= count0[a] #find the average of single bins with multiple velocity values

    plot_vel[:, column] = vel0 #add velocity meridians into array to be plotted

    velocity_times.append(rtime) #add the current time to the total time length array

    dfVel.read_record()
    if not isinstance(dfVel.dstr,read_df_record.record):
        break

    column += 1 #step fowards in the column

nonzero_rows = np.any(plot_vel != 0, axis=1) #find all rows full of zeros
nonzero_cols = np.any(plot_vel != 0, axis=0) #find all columns full of zeros

plot_vel = plot_vel[nonzero_rows][:, nonzero_cols] #remove the rows and columns full of zeros from the array
# ^this is the final velocity array for the given MLT

#==========================================================================================================================================
#==========================================================================================================================================
#make a grid of the latitudes
column_stack = len(plot_vel[0, :])
latitudes_column = latitudes.reshape(-1, 1)
lat_grid = np.tile(latitudes_column, (column_stack))

#==========================================================================================================================================
#==========================================================================================================================================
#break brightness data up into two minute intervals to match with velocity
split_size = n_intervals/len(velocity_times) #splits the total time intervals into two minute segments

img_splits = plot_img2.reshape(plot_img2.shape[0], -1, int(split_size)) #split the brightness array every two minutes
plot_img2 = np.mean(img_splits, axis = 2) #take the average of each split and put it into a new array

num_columns = len(plot_img2[0, :]) #number of columns; equal to number of time points
num_rows = len(plot_img2[:, 0]) #number of rows; equal to number of latitude points in latitude range

#==========================================================================================================================================
#==========================================================================================================================================
#run 10 minute correlation blocks for brightness and velocity
plot_corr = np.zeros((num_rows, num_columns))
correlation_times = []
for i in velocity_times:
    correlation_times.append(i)

corr_ndtime = correlation_times[-1] + timedelta(minutes = 8) #initial correlation end time; 10 minutes after velocity ends due to 10 minute correlation intervals
add_from = correlation_times[-1]

while add_from <= corr_ndtime:
    add_from += timedelta(minutes = 2)
    correlation_times.append(add_from)

total_points = len(correlation_times)
#print(correlation_times[-6])

print('\nCorrelation')
for i in range(total_points):
    try:
        st_time = correlation_times[i] #start time
        nd_time = correlation_times[i+5] #end time; 10 minutes after start time

        if nd_time <= correlation_times[-1]:
            print(str(st_time)+' to '+str(nd_time))
                
            for j in range(num_rows): #loop through rows
                bright_values = plot_img2[j, i:i+5]
                vel_values = plot_vel[j, i:i+5]

                #correlation
                #term1 = 1/len(vel_values) * np.sum( (vel_values - np.mean(plot_vel)) * (bright_values - np.mean(plot_img2)) )
                #term2 = np.sqrt(1/len(vel_values) * np.sum(vel_values - np.mean(plot_vel))**2)
                #term3 = np.sqrt(1/len(vel_values) * np.sum(bright_values - np.mean(plot_img2))**2)

                term1 = len(vel_values) * np.sum(vel_values*bright_values) - np.sum(vel_values) * np.sum(bright_values)
                term2 = len(vel_values) * np.sum(vel_values**2) - np.sum(vel_values)**2
                term3 = len(vel_values) * np.sum(bright_values**2) - np.sum(bright_values)**2
                r2 = ( term1/(np.sqrt(term2) * np.sqrt(term3)) ) #calculation for correlation coefficient

                #print(term2)
                #print(term3)

                if corr:
                    plot_corr[j, i:i+5] = r2

                #covariance

                else:
                    covar = 1/len(vel_values) * np.sum( (vel_values - np.mean(plot_vel)) * (bright_values - np.mean(plot_img2)) )

                    print('velocity:', vel_values)
                    print('brightness:', bright_values)

                    print(np.mean(plot_vel))
                    print(np.mean(plot_img2), '\n')

                    plot_corr[j, i:i+5] = covar

    except IndexError:
        break

#==========================================================================================================================================
#==========================================================================================================================================
#mask for image data
mask = plot_img2 == 0
plot_img2 = np.ma.masked_where(mask, plot_img2)

#create multiplot for brightness, velocity, and correlation
times = velocity_times
subplots = 3
fig, axs = plt.subplots(subplots, figsize = (14, 22), sharex = True, layout = 'constrained')

locator = mdates.AutoDateLocator(minticks = 3, maxticks = 12)
formatter = mdates.ConciseDateFormatter(locator)

#balance colorbar for covariance plot
max_cov = np.max(plot_corr)
min_cov = np.min(plot_corr)

if max_cov > np.abs(min_cov) or max_cov == np.abs(min_cov):
    vmax = max_cov
    vmin = -max_cov

elif max_cov < np.abs(min_cov): 
    vmax = np.abs(min_cov)
    vmin = min_cov

#find latitudes of minimum and maximum covariance values
flat_index = np.argmax(plot_corr)
y_index, x_index = np.unravel_index(flat_index, plot_corr.shape)
max_lat = lat_grid[y_index, 0]

flat_index = np.argmin(plot_corr)
y_index, x_index = np.unravel_index(flat_index, plot_corr.shape)
min_lat = lat_grid[y_index, 0]

#plot velocity on first plot
x = times
y = lat_grid
z = plot_vel

axs[0].xaxis.set_major_locator(locator)
axs[0].xaxis.set_major_formatter(formatter)
plot = axs[0].pcolormesh(x, y, z, cmap = 'plasma')
axs[0].set_ylabel('Latitude (Degrees)', fontsize = 25)
axs[0].tick_params(labelsize = 20)

cbar = fig.colorbar(plot, ax=axs[0], shrink=0.9)
cbar.ax.set_ylabel('Velocity Magnitude (m/s)', fontsize = 25)
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_fontsize(20)

#plot lines across plot
lines = 6 #number of lines to plot across figure
indicies = np.linspace(0, len(lat_grid[:, 0]) - 1, lines + 2, dtype = int)
line_points = lat_grid[indicies, 0]
line_points = line_points[1:-1]

if fig_lines:
    axs[0].axhline(y = max_lat, color = 'black', linewidth = 6, linestyle = 'dashed', label = 'Latitude of maximum covariance')
    axs[0].axhline(y = min_lat, color = 'black', linewidth = 6, linestyle = 'dashdot', label = 'Latitude of minimum covariance')
    axs[0].legend(prop={'size': 15})

#plot brightness on second plot
#x = ptimes[0:len(img02), :]
x = times
y = lat_grid
z = plot_img2

axs[1].xaxis.set_major_locator(locator)
axs[1].xaxis.set_major_formatter(formatter)
plot = axs[1].pcolormesh(x, y, z, cmap = cmap)
axs[1].set_ylabel('Latitude (Degrees)', fontsize = 25)
axs[1].tick_params(labelsize = 20)

cbar = fig.colorbar(plot, ax=axs[1], shrink=0.9)
cbar.ax.set_ylabel('Brightness', fontsize = 25)
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_fontsize(20)

#plot lines across figure
if fig_lines:
    axs[1].axhline(y = max_lat, color = 'black', linewidth = 6, linestyle = 'dashed', label = 'Latitude of maximum covariance')
    axs[1].axhline(y = min_lat, color = 'black', linewidth = 6, linestyle = 'dashdot', label = 'Latitude of minimum covariance')
    axs[1].legend(prop={'size': 15})

#plot covariance on third plot
x = times
y = lat_grid
z = plot_corr

axs[2].xaxis.set_major_locator(locator)
axs[2].xaxis.set_major_formatter(formatter)
plot = axs[2].pcolormesh(x, y, z, cmap = 'RdYlGn')
axs[2].set_ylabel('Latitude (Degrees)', fontsize = 25)
axs[2].set_xlabel('Time (UT)', fontsize = 25)
axs[2].tick_params(labelsize = 20)
plt.xticks(fontsize = 20)

cbar = fig.colorbar(plot, ax=axs[2], shrink=0.9)

if corr:
    cbar.ax.set_ylabel('Correlation', fontsize = 25)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontsize(20)

else:
    cbar.ax.set_ylabel('Covariance', fontsize = 25)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontsize(20)

    plot.set_clim(vmin, vmax)

#plot lines across figure
if fig_lines:
    axs[2].axhline(y = max_lat, color = 'black', linewidth = 6, linestyle = 'dashed', label = 'Latitude of maximum covariance')
    axs[2].axhline(y = min_lat, color = 'black', linewidth = 6, linestyle = 'dashdot', label = 'Latitude of minimum covariance')
    axs[2].legend(prop={'size': 15})
#save the figure
fig.suptitle(str(int(mlt))+' MLT', fontsize = 30, weight = 'bold')

if corr and not fig_lines:
    plt.savefig(str(int(mlt))+'MLT_correlation_tser.png')

if corr and fig_lines:
    plt.savefig(str(int(mlt))+'MLT_correlation_tser_line.png')

elif not fig_lines:
    plt.savefig(str(int(mlt))+'MLT_covariance_tser.png')

elif fig_lines:
    plt.savefig(str(int(mlt))+'MLT_covariance_tser_line.png')

print('\n'+str(int(mlt)), 'MLT')
print('Plot has been made')