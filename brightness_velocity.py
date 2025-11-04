#!/usr/bin/env python3

#THIS SCRIPT GETS BOTH BRIGHTNESS AVERAGES AND VELOCITY AVERAGES OVER A LATITUDE RANGE
#TO CALL THE SCRIPT: MLT file_name starttime endtime (MLT vel_out 201403030000 201403032359)

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
import julian
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

import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
parser.add_argument('fname', type=str) #file name
parser.add_argument("sttime", type=str) #start time argument; YYYYmmddHHMM
parser.add_argument("ndtime", type=str) #end time argument; YYYYmmddHHMM

#parser.add_argument('--limit1', required=False, type=str, default=None) #x-axis lower limit for time series plot; YYYYmmddHHMM
#parser.add_argument('--limit2', required=False, type=str, default=None) #x-axis upper limit for time series plot: YYYmmddHHMM

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
#limit1 = args.limit1
#limit2 = args.limit2

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
saved_times = [] #empty array to hold times where there are velocities
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

    plat = 70 #latitude the range is centered around
    dlat = 0.5 #latitude range below and above center

    lat_inds = (lats > plat-dlat) & (lats < plat+dlat) #makes an array of True and False that defines the latitude range

    x_vel = np.sum(vx_marid[lat_inds])/len(vx_marid[lat_inds]) #average x velocity over defined latitude range
    y_vel = np.sum(vy_marid[lat_inds])/len(vy_marid[lat_inds]) #average y velocity over defined latitude range

    vx.append(x_vel)
    vy.append(y_vel)

    saved_times.append(rtime)

    dfVel.read_record()
    if not isinstance(dfVel.dstr,read_df_record.record):
        break

#==========================================================================================================================================
#calculate total velocity from x and y velocities
v_totals = []
for i in range(len(vx)):
    vel = np.sqrt(vx[i]**2 + vy[i]**2)
    v_totals.append(vel)

print('Total Velocity:\n',np.round(v_totals, 5))
print(len(v_totals))


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
#sites=['kian','atha','fykn','inuv','whit','gako','fsmi','fsim','gill','rank','kuuj','snkq'] #every site #multiple sites
sites = ['whit'] #single site
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


#loop through sites and print data
img_avgs = [] #empty array to hold brightness averages over entire time series
saved_times2 = [] #empty array to hold times where there is a brightness
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
                saved_times2.append(time) #adds the current times to an array to be plotted
            except:
                print('Division by Zero')
                continue

#==========================================================================================================================================
#plot velocity and brigtness averages on same plot
xdm = 20
ydm = 12
fsz = 'large'

def capitalize_list(item): #define function to capitalize site names
    return item.upper()

SITES = (list(map(capitalize_list, sites))) #capitalize site names in site list
#print(sites)
#print(SITES)

fig, ax = plt.subplots(2, figsize = (xdm, ydm))
fig.suptitle(str(mlt).rstrip('.0')+' MLT Velocity and Brightness Averages from '+sttime+' to '+ndtime)

locator = mdates.AutoDateLocator(minticks = 3, maxticks = 12)
formatter = mdates.ConciseDateFormatter(locator)

ax[0].xaxis.set_major_locator(locator)
ax[0].xaxis.set_major_formatter(formatter)
ax[1].xaxis.set_major_locator(locator)
ax[1].xaxis.set_major_formatter(formatter)

x1 = saved_times #time array for total velocity averages
x2 = saved_times2 #time array for brightness averages
y1 = v_totals #array for total velocity averages for each time step
y2 = img_avgs #array of brightness averages for each time step

ax[0].plot(x1, y1)
ax[1].plot(x2, y2)

ax[0].set_xlabel('Time (UT)')
ax[1].set_xlabel('Time (UT)')
ax[0].set_ylabel('Velocity (m/s)')
ax[1].set_ylabel('Brightness From '+SITES[0])

limit1 = (saved_times2[0]).strftime('%Y-%m-%d %H:%M:%S')
limit2 = (saved_times2[-1]).strftime('%Y-%m-%d %H:%M:%S')

limit1 = datetime.strptime(limit1, '%Y-%m-%d %H:%M:%S')
limit2 = datetime.strptime(limit2, '%Y-%m-%d %H:%M:%S')

ax[0].set_xlim(mdates.date2num(limit1), mdates.date2num(limit2))
ax[1].set_xlim(mdates.date2num(limit1), mdates.date2num(limit2))

pname = 'brightness_velocity_avg.png'
plt.savefig(pname)
print('Plot has been made')
#print(saved_times2)
