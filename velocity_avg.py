#!/usr/bin/env python3

#THIS SCRIPT GETS AN AVERAGE VELOCITY OF THE PLASMA OVER A LATITUDE RANGE
#TO CALL THE SCRIPT: File Name, MLT (vel_out 22)
#TO ADD START TIME AND END TIME: --sttime 201403030900 --ndtime 201403031100

#import required modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import argparse
import julian
import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.backends.backend_pdf import PdfPages

# local import
from df_vel_class import df_vel
from df_pot_class import df_pot
from date_strings import make_date_str, make_date_time_str, cnv_datetimestr_dtlist
from mag_continents import magContinents
import aacgm
import read_df_record

from themis_mlt_keogram import KeogramOverlay
#from pwr_col_map import pwr_col
#keo_cmap=pwr_col()

parser = argparse.ArgumentParser(description='local_df_tser argument parser')
parser.add_argument('fname', type=str) #file name
parser.add_argument('mlt', type=float) #magnetic time
parser.add_argument("--sttime", nargs=1, type=str, required=False, default=190001010000,
                    help="sttime must be specified (eg: '201001200000')")
parser.add_argument("--ndtime", nargs=1, type=str, required=False, default=299912312359,
                    help="ndtime must be specified (eg: '201001202359')")
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

args = parser.parse_args()
fname=args.fname
mlt=args.mlt
sttime=args.sttime
ndtime=args.ndtime
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


dfVel=df_vel(fname,global_vel=global_v)

if pot_file != None:
    dfPot=df_pot(fname=pot_file[0])


if mag:
    aacgm.set_datetime(dfVel.dstr.yr,dfVel.dstr.mo,dfVel.dstr.dy,int(dfVel.dstr.hr),int(dfVel.dstr.mt),0)

    for j in range(dfVel.dstr.num):
        pos=aacgm.convert(dfVel.dstr.vec_lat[j],dfVel.dstr.vec_lon[j],300,0)
        dfVel.dstr.vec_lat[j]=pos[0]
        dfVel.dstr.vec_lon[j]=pos[1]

xdim=14
ydim=9
fig=plt.figure(figsize=(xdim, ydim))
ax = fig.add_subplot()


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
        #print(lat)
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

#print(np.round(vx, 2), np.round(vy, 2))
#print(len(vx), len(vy))
#print(len(saved_times))

v_totals = []
for i in range(len(vx)):
    vel = np.sqrt(vx[i]**2 + vy[i]**2)
    v_totals.append(vel)

print(np.round(v_totals, 5))
print(len(v_totals))

xdm = 20
ydm = 9
fig = plt.figure(figsize = (xdm, ydm))
ax = fig.add_subplot()

locator = mdates.AutoDateLocator(minticks=3, maxticks=12)
formatter = mdates.ConciseDateFormatter(locator)

ax.plot(saved_times, v_totals)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.set_xlabel('Time (UT)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title(str(mlt)+' MLT '+sttime+'-'+ndtime+' Velocity')

pname = sttime+'-'+ndtime+'velocity_totals.png'
pname = 'velocity_totals.png'
plt.savefig(pname)
print('Plot has been made')
