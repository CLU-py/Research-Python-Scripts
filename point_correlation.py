#!/usr/bin/env python3

#THIS SCRIPT GETS BOTH BRIGHTNESS AVERAGES AND VELOCITY AVERAGES OVER A LATITUDE RANGE
#TO CALL THE SCRIPT: MLT plat file_name starttime endtime (23 68 vel_out 201403030000 201403032359)
#TO CALL WITH CORRELATION: MLT plat file_name starttime endtime --corr --correlation_sttime sttime --correlation_ndtime ndtime

#==========================================================================================================================================
#import required modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import argparse
import os
import scipy.io as sio
import glob
import jlian
import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from matplotlib import colors as mpl_colors
from matplotlib.colors import Normalize

import cdflib
from cdflib import cdfread
from cdflib.xarray import cdf_to_xarray

from df_vel_class import df_vel
from df_pot_class import df_pot
from date_strings import make_date_str, make_date_time_str, cnv_datetimestr_dtlist
from mag_continents import magContinents
import aacgm
import read_df_record

from themis_mlt_keogram import KeogramOverlay

from map_utils import min_sep

from pwr_col_cmap import pwr_col
cmap = pwr_col()

import time as tmr

import aacgm
from date_strings import make_date_str, make_date_time_str, cnv_datetimestr_dtlist, cnv_datetimestr_datetime

#==========================================================================================================================================
#add commands to call in terminal when calling the script; see top of script for command order
parser = argparseparser = argparse.ArgumentParser(description='sector_gdf_vel argument parser')
parser.add_argument('plot_mlt', type=float) #argument of MLT; write as integer betweem 00 and 23
parser.add_argument('plat', type=float) #planetary latitude the latitude bin will be centered around
parser.add_argument('dlat', type=float) #planeraty latitude range around the centered latitude (plat)
parser.add_argument('fname', type=str) #file name
parser.add_argument("sttime", type=str) #start time argument; YYYYmmddHHMM
parser.add_argument("ndtime", type=str) #end time argument; YYYYmmddHHMM

parser.add_argument('--corr', required=False, dest='corr', action='store_true', default=False) #argument which tells the script to run the correlation
parser.add_argument('--timelag', required=False, type=float, default=0) #argument for the time lag in minutes when correlating brightness and velocity

parser.add_argument("--pot_file", required=False, nargs=1, type=str, default=None)
parser.add_argument('--pot', required=False, dest='pot', action='store_true', default=False)
parser.add_argument('--mag', required=False, dest='mag', action='store_true', default=False)
parser.add_argument('--mxval', required=False, type=int, default=20000)
parser.add_argument('--scale', required=False, type=int, default=1000)
parser.add_argument('--fitvecs', required=False, dest='fitvecs', action='store_true', default=False)
parser.add_argument('--no_png', required=False, dest='png', action='store_false', default=True)
parser.add_argument('--global', required=False, dest='global_v', action='store_true', default=False)
parser.add_argument('--min_lat', dest='in_min_lat', required=False, type=float, default=None)
parser.add_argument('--max_lat', dest='in_max_lat', required=False, type=float, default=None)
parser.add_argument('--dlat', dest='dlat', required=False, type=float, default=.5)
parser.add_argument('--cvecs', required=False, dest='cvecs', action='store_true', default=False)
parser.add_argument('--asi_keo', required=False, dest='asi_keo', action='store_true', default=False)
parser.add_argument("--cmap_name", required=False, type=str, default=None)
parser.add_argument('--lat_min', required=False, type=float, default=60) #minimum latitude, optional
parser.add_argument('--lat_max', required=False, type=float, default=80) #maximum latitude, optional

args = parser.parse_args()
fname = args.fname
plot_mlt = args.plot_mlt
mlt = plot_mlt
sttime = args.sttime
ndtime = args.ndtime
plat = args.plat
dlat_bin = args.dlat

corr = args.corr
time_lag = args.timelag

mxval=args.mxval
pot=args.pot
mag=args.mag
scale=args.scale
pot_file = args.pot_file
fitvecs=args.fitvecs
png=args.png
global_v=args.global_v
in_min_lat=args.in_min_lat
in_max_lat=args.in_max_lat
dlat=args.dlat
cvecs=args.cvecs
asi_keo=args.asi_keo
cmap_name=args.cmap_name
lat_min=args.lat_min
lat_max=args.lat_max

#==========================================================================================================================================
#get velocity data
dfVel=df_vel(fname,global_vel=global_v)

if pot_file != None:
    dfPot=df_pot(fname=pot_file[0])

if mag:
    aacgm.set_datetime(dfVel.dstr.yr,dfVel.dstr.mo,dfVel.dstr.dy,int(dfVel.dstr.hr),int(dfVel.dstr.mt),0)

    for j in range(dfVel.dstr.num):
        pos=aacgm.convert(dfVel.dstr.vec_lat[j],dfVel.dstr.vec_lon[j],300,0)
        dfVel.dstr.vec_lat[j]=pos[0]
        dfVel.dstr.vec_lon[j]=pos[1]

dtor=np.pi/180
inds=[]
dlon=.75
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

entry = True

#==========================================================================================================================================
#calculte x and y velocities
velocity_times = [] #empty array to hold times where there are velocities
vx = [] #empty array to hold x velocities
vy = [] #empty array to hold y velocities

while 1:

    vel_x[:] = 0.0
    vel_y[:] = 0.0
    count_ar[:] = 0.0

    rtime = datetime.datetime(dfVel.dstr.yr, dfVel.dstr.mo, dfVel.dstr.dy,
                              dfVel.dstr.hr, dfVel.dstr.mt, int(dfVel.dstr.sc))

    print(rtime)
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

    vx_marid = dfVel.dstr.vx[inds] #x-velocities across entire latitude range (55 to 88)
    vy_marid = dfVel.dstr.vy[inds] #y-velocities across entire latitude range (55 to 88)
    lats = dfVel.dstr.vec_lat[inds] #full range of latituides from 55 to 88

    #plat = 70 #latitude the range is centered around
    #dlat = 0.5 #latitude range below and above center

    lat_inds = (lats > plat-dlat_bin) & (lats < plat+dlat_bin) #makes an array of True and False that defines the latitude range

    x_vel = np.sum(vx_marid[lat_inds])/len(vx_marid[lat_inds]) #average x velocity over defined latitude range
    y_vel = np.sum(vy_marid[lat_inds])/len(vy_marid[lat_inds]) #average y velocity over defined latitude range

    vx.append(x_vel)
    vy.append(y_vel)

    velocity_times.append(rtime)

    dfVel.read_record()
    if not isinstance(dfVel.dstr,read_df_record.record):
        break

#========================================================================================================================================++
#calculate total velocity from x and y velocities
v_totals = []
for i in range(len(vx)):
    vel = np.sqrt(vx[i]**2 + vy[i]**2)
    v_totals.append(vel)

print('\nTotal Velocity:\n',np.round(v_totals, 5))
#print(len(v_totals))

#==========================================================================================================================================
#==========================================================================================================================================
import datetime as dt
from datetime import datetime
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
sites=['kian','atha','fykn','inuv','whit','gako','fsmi','fsim','gill','rank','kuuj','snkq'] #every site
scales={'atha':2000,'fsim':4500,'fykn':2000,'inuv':2000,'whit':2500,'fsmi':5000,'gako':2000,'gill':2500,'rank':4000,'kian':3000, 'snkq':4000}

n_hours=int(np.rint((end_time-start_time).seconds/3600)) #number of hours

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
#loop through sites and print data
i = 0 #value corresponding to subplot axis to be plotted on; increases after each iteration of loop
brightness_list = [[] for _ in range(len(sites))] #empty array for brightness from all sites that have data in selected location and time range
times_list = [[] for _ in range(len(sites))] #empty array for times where have data in selected location and time range
sites_list = [[] for _ in range(len(sites))] #empty array for sites that have data in selected location and time range
for site in sites:
    print('\n', site)
    img_avgs = [] #empty array to hold brightness averages over entire time series
    brightness_times = [] #empty array to hold times where there is a brightness

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
            #print(site)

            finelons=np.isfinite(lons)
            neglons=lons<0
            lons[neglons]+=360

            pixls=(lons > plon-dlon) & (lons < plon+dlon) & (elev > 5)
            px_lats=lats[pixls] #latitudes of the pixels in the image

            img_pixls = image_data[pixls] - 2600 #brightness value of aurora
            #plat = 70 #latitude for which the range is centered around
            #dlat = .5 #change in latitude used to define range
            pixls2 = (px_lats > plat-dlat_bin) & (px_lats < plat + dlat_bin) #pixels within the latitude range

            #print(px_lats) #prints the latitudes of the pixels in the image
            #print(img_pixls) #prints the array of brightness values
            #print(px_lats[pixls2], img_pixls[pixls2]) #prints the brigtness values in the latitude range

            try:
                img_avg = sum(img_pixls[pixls2]) / len(img_pixls[pixls2]) #averages the brightness values
                print('Brightness Average:', img_avg) #prints a single brightness value representing the average over the latitude range
                img_avgs.append(img_avg) #adds the brightness average for a single time to an array to hold all averages over the time series
                brightness_times.append(time) #adds the current times to an array to be plotted
            except:
                print('Division by Zero')
                continue

    if not img_avgs:
        print('No site data for this time at '+site)
    else:
        brightness_list[i] = img_avgs
        times_list[i] = brightness_times
        sites_list[i] = site

        i += 1

#remove empty elements from the brightness, time, and sites array
brightness_list = [sublist for sublist in brightness_list if sublist]
times_list = [sublist for sublist in times_list if sublist]
sites_list = [sublist for sublist in sites_list if sublist]

def capitalize_list(item): #define function to capitalize site names
    return item.upper()

sites_list = (list(map(capitalize_list, sites_list))) #capitalize site names in site list for plotting

#==========================================================================================================================================
#create figure with subplots based on the number of sites being used
subplots = int(len(sites_list))
fig, axs = plt.subplots(subplots+1, figsize = (14, 22), sharex = True, layout = 'constrained')

locator = mdates.AutoDateLocator(minticks = 3, maxticks = 12)
formatter = mdates.ConciseDateFormatter(locator)

#plot velocity on first plot
axs[0].xaxis.set_major_locator(locator)
axs[0].xaxis.set_major_formatter(formatter)
axs[0].plot(velocity_times, v_totals)
axs[0].set_ylabel('Velocity (m/s')

fig.suptitle(str(int(float(plot_mlt)))+'MLT Brightness and Velocity Averages from '+sttime+' to '+ndtime+' at '+str(plat-dlat_bin)+' to '+str(plat+dlat_bin)+' Degrees Latitude', fontsize = 15, weight='bold')
fig.supxlabel('Time (UT)', fontsize = 15, weight='bold')

for k in range(len(sites_list)):
    axs[k+1].xaxis.set_major_locator(locator)
    axs[k+1].xaxis.set_major_formatter(formatter)

    axs[k+1].plot(times_list[k], brightness_list[k], color = 'red')
    axs[k+1].set_ylabel('Brightness at '+sites_list[k])

pname = 'brightness_velocity_multi.png'
plt.savefig(pname)
print('Plot has been made\n')


#==========================================================================================================================================
#run correlation between brightness and velocity
from datetime import datetime, timedelta

if corr:
    corr_site = input('List of sites with data for this time are '+str(sites_list)+'\nWhich site would you like to correlate with velocity? ')
    for s in range(len(sites_list)):
        if sites_list[s] == corr_site:
            list_index = s
            #print(list_index)

    brightness = brightness_list[list_index]
    velocity = v_totals
    #print(np.round(brightness, 2))

    #average brightness data over 1 minute intervals
    interval_seconds = 120 #two minute interval to average over

    brightness_times_min = []
    brightness_min = []
    start_time = times_list[list_index][0]
    if (start_time.minute % 2) == 0:
        start_time = datetime(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute+1) #make the starting minute odd to line up with velocity times
    else:
        start_time = datetime(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute) #leave the starting minute odd to line up with velocity times
    sum_values = 0
    count_values = 0
    for i, dt in enumerate(times_list[list_index]):
        sum_values += brightness[i]
        count_values += 1
        if (dt-start_time).total_seconds() >= interval_seconds:
            brightness_avg = sum_values/count_values
            brightness_min.append(brightness_avg)
            brightness_times_min.append(start_time)
            
            #reset for the next two minute
            start_time = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
            sum_values = 0
            count_values = 0

    brightness_times_min.append(start_time)

    #print('original brightness times: ',brightness_times,'\n')
    #print('new brightness times: ',brightness_times_min,'\n')

    #find what index in the velocity times matches up with the initial brightness times
    lag_index = int(time_lag//2)

    vel_stindex = velocity_times.index(brightness_times_min[0+lag_index])
    vel_ndindex = velocity_times.index(brightness_times_min[5+lag_index])

    bright_stindex = brightness_times_min.index(brightness_times_min[0])
    bright_ndindex = brightness_times_min.index(brightness_times_min[5])

    #run correlation in 10 minute intervals with one minute steps 
    sttime = brightness_times_min[0]
    ndtime = brightness_times_min[-1]

    if time_lag != 0:
        print('\nRunning Correlation from', str(sttime), 'to', str(ndtime), 'at Site', str(corr_site), 'With', str(int(float(time_lag))), 'Minutes of Time Lag')
    else:
        print('\nRunning Correlation from', str(sttime), 'to', str(ndtime), 'at Site', str(corr_site))

    velocity = np.array(velocity)
    brightness_min = np.array(brightness_min)

    t = 0
    midpoint_times = []
    start_times = []
    coeff_det = []
    for i in range(1000):
        sttime = brightness_times_min[t]
        ndtime = brightness_times_min[t+5]

        if ndtime <= brightness_times_min[-1]:
            
            try:
                vel_values = velocity[vel_stindex:vel_ndindex]
                bright_values = brightness_min[bright_stindex:bright_ndindex]

                term1 = len(vel_values) * np.sum(vel_values*bright_values) - np.sum(vel_values) * np.sum(bright_values)
                term2 = len(vel_values) * np.sum(vel_values**2) - np.sum(vel_values)**2
                term3 = len(vel_values) * np.sum(bright_values**2) - np.sum(bright_values)**2
                r2 = ( term1/(np.sqrt(term2) * np.sqrt(term3)) )**2
                print('Coefficient of Determination From',sttime, 'to', ndtime,':',round(r2, 2))

                vel_stindex += 1
                vel_ndindex += 1
                bright_stindex += 1
                bright_ndindex += 1
                #print('velocity index:',vel_stindex)
                #print('brightness index:',bright_stindex)

                diff = ndtime - sttime #difference between the start and end time
                midpoint_time = sttime + diff/2 #calculates the time in between the current start and end time
                midpoint_times.append(midpoint_time) #appends the time in between the current start and end time to an array

                start_times.append(sttime) #appends the current start time to an array
                t += 1
                sttime = brightness_times_min[t] #makes the new start time the next time in the array
                coeff_det.append(r2) #appends the current coefficient of determination to an array

                try:
                    ndtime = brightness_times_min[t+5]
                except IndexError:
                    break
            except ValueError:
                break

        else:
            break
    
    #create triple box figure for brightness, velocity, and correlation
    subplots = 3
    fig, axs = plt.subplots(subplots, figsize = (14, 22), sharex = True, layout = 'constrained')    

    #plot velocity on first plot
    axs[0].xaxis.set_major_locator(locator)
    axs[0].xaxis.set_major_formatter(formatter)
    axs[0].plot(velocity_times, v_totals)
    axs[0].set_ylabel('Velocity (m/s')
    axs[0].set_xlim(times_list[list_index][0], times_list[list_index][-1])

    #plot brightness on second plot
    axs[1].xaxis.set_major_locator(locator)
    axs[1].xaxis.set_major_formatter(formatter)
    axs[1].plot(times_list[list_index], brightness, color='red')
    axs[1].set_ylabel('Brightness at'+corr_site)
    axs[1].set_xlim(times_list[list_index][0], times_list[list_index][-1])

    #plot correlation on third plot
    axs[2].xaxis.set_major_locator(locator)
    axs[2].xaxis.set_major_formatter(formatter)
    axs[2].plot(start_times, coeff_det, color='green')
    axs[2].set_ylabel('Coeff of Determination')
    axs[2].set_xlim(times_list[list_index][0], times_list[list_index][-1])

    sttime = brightness_times_min[0].strftime("%Y%m%d%H%M")
    ndtime = brightness_times_min[-1].strftime("%Y%m%d%H%M")

    if time_lag != 0:
        fig.suptitle(str(int(float(plot_mlt)))+'MLT Correlation of Brightness and Velocity from '+str(sttime)+' to '+str(ndtime)+' at Site '+str(corr_site)+'\nWith '
                +str(int(float(time_lag)))+' Minutes of Time Lag', fontsize = 15, weight='bold')
    else:
        fig.suptitle('Correlation of Brightness and Velocity from '+str(sttime)+' to '+str(ndtime)+' at Site '+str(corr_site), fontsize = 15, weight='bold')
    fig.supxlabel('Time (UT)', fontsize = 15, weight='bold')

    pname = 'correlation_plot.png'
    plt.savefig(pname)
    print('Plot has been made\n')
