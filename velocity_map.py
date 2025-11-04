#!/usr/bin/env python3

#THIS SCRIPT GETS THE VELOCITY AT ALL MERIDIANS WITH A ONE DEGREE LATITUDE BIN AND PLOTS IT ON A HEMISPHERICAL MAP
#TO CALL THIS SCRIPT: fileName starttime endtime (vel_out 201403210400 201403210600)
#IF SCRIPT IS NOT EXECUTABLE IN TERNIMAL, RUN chmod +x {filename}.py

#TO RUN PROCESS IN SCREEN MODE: 'screen' TO START SCREEN SESSION, crtl + a (release), then d TO DETACH THE PROCESS/SCREEN
#TO RESUME DETACHED PROCESS: screen -r IN TERMINAL

#TO SEE PROCESSES RUNNING IN TERMINAL: 'top' IN COMMAND LINE

#==========================================================================================================================================
#==========================================================================================================================================
#import required modules
import numpy as np
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import datetime as dt
import aacgm
import read_df_record
import cartopy .crs as ccrs

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from map_grid import make_grid #refer to map_grid.py
from df_vel_class import df_vel #dataframe for velocity
from date_strings import make_date_str, make_date_time_str, cnv_datetimestr_dtlist, cnv_datetimestr_datetime
#from mag_continents import magContinents

#==========================================================================================================================================
#==========================================================================================================================================
#and commands to call in terminal when running script
parser = argparseparser = argparse.ArgumentParser(description='sector_gdf_vel argument parser')
parser.add_argument('fname', type=str) #file name (this is just vel_out)
parser.add_argument('sttime', type=str) #start time argument; YYYmmddHHMM
parser.add_argument('ndtime', type=str) #end time argument; YYYmmddHHMM
parser.add_argument('--meridians', type = float, nargs = "+", default = None)

args = parser.parse_args()
fname = args.fname
sttime = args.sttime
ndtime = args.ndtime
meridians = args.meridians

mag = True #if True, script is using magnetic time

#==========================================================================================================================================
#==========================================================================================================================================
#define function to plot meridian lines
def cnv_mlt_label(mlt):
    mlt_hour=int(mlt)
    mlt_min=int(60*(mlt-mlt_hour))

    return(str(mlt_hour).rjust(2,'0')+str(mlt_min).rjust(2,'0')+" MLT")

#==========================================================================================================================================
#==========================================================================================================================================
#make grid to project onto map
grid = make_grid() #call make_grid function from map_grid.py file

#==========================================================================================================================================
#==========================================================================================================================================
#get velocity values from vel_out file
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
print('Get velocity values:')
while 1:
    rtime = dt.datetime(dfVel.dstr.yr, dfVel.dstr.mo, dfVel.dstr.dy,
                              dfVel.dstr.hr, dfVel.dstr.mt, int(dfVel.dstr.sc))

    print(rtime) #current time velocity data is being taken from
    #print(dfVel.dstr.yr, dfVel.dstr.mo, dfVel.dstr.dy, int(dfVelplasma.dstr.hr), int(dfVel.dstr.mt))
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
    #define corners of every grid box from full map grid
    lat_ll = grid['lat_ll'] #list of lower left latitudes
    lat_ul = grid['lat_ul'] #list of upper left latitudes

    lon_ll = grid['lon_ll'] #list of lower left longitudes
    lon_lr = grid['lon_lr'] #list of lower right longitudes

    vel_grid = np.zeros(len(grid['index'])) #empty list for velocity grid where radar data was used
    vel_grid_nd = np.zeros(len(grid['index'])) #empty list for velocity grid where radar data was not used

    #======================================================================================================================================
    #place velocity values into empty grid
    for i in range(len(grid['index'])): #loop through the grid cells
        count = 0 #increases when velocity values fall into same grid box
        count_nd = 0
        for j in range(len(v_mag)): #loop through velocity values
            #print(dfVel.dstr.dcount[j])
            if dfVel.dstr.dcount[j] != 0: #tells if the velocity value considered data from radars
                if lat_ll[i] < v_lats[j] and lat_ul[i] > v_lats[j]:
                    if lon_ll[i] < v_lons[j] and lon_lr[i] > v_lons[j]:
                        vel_grid[i] += v_mag[j]
                        count += 1

            else: #velocity value did not consider data from radar
                if lat_ll[i] < v_lats[j] and lat_ul[i] > v_lats[j]:
                    if lon_ll[i] < v_lons[j] and lon_lr[i] > v_lons[j]:
                        vel_grid_nd[i] += v_mag[j]
                        count_nd += 1

        vel_grid[i] = vel_grid[i]/count
        vel_grid_nd[i] = vel_grid_nd[i]/count_nd

    plot_time = make_date_time_str(dfVel.dstr) #current time data is being collected from

    #======================================================================================================================================
    #======================================================================================================================================
    #map grid to plot and plot velocity values
    fg=open('grid.dat','r')
    min_lat=float(fg.readline().split()[1])
    max_lat=float(fg.readline().split()[1])
    min_lon=float(fg.readline().split()[1])
    max_lon=float(fg.readline().split()[1])

    #======================================================================================================================================
    #create a cirlce overlay for the plot
    theta = np.linspace(0, 2*np.pi, 360)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T

    circle = mpath.Path(verts * radius + center)

    #======================================================================================================================================
    #set a longitude aligned with noon and a latitude for pole
    noonlon=aacgm.inv_mlt_convert(dfVel.dstr.yr, dfVel.dstr.mo, dfVel.dstr.dy, int(dfVel.dstr.hr), int(dfVel.dstr.mt)-2, 0, 12)-180
    #print(noonlon)
    pole_lat = 90

    #======================================================================================================================================
    #create figure
    fig=plt.figure(figsize=(10, 10), dpi=200)

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

    lab_lat = min_lat+2
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

    xpos=.05
    ypos=.97
    ax.annotate(plot_time, (xpos,ypos), xycoords='axes fraction')

    #======================================================================================================================================
    #======================================================================================================================================
    #create patches, add them to the plot, and add a colorbar
    colormap = mpl.colormaps['plasma'] #get colormap object from matplotlib colormap
    norm = Normalize(vmax = 1500, vmin = 0)

    patches = [] #patches where radar data was used
    patches_nd = [] #patches where radar data was not used

    #======================================================================================================================================
    #create patches where radar data was not used
    for i in range(len(grid['index'])):
        x = lon_ll[i]
        y = lat_ll[i]
        width = lon_lr[i] - lon_ll[i]
        height = lat_ul[i] - lat_ll[i]

        color = colormap(norm(vel_grid_nd[i]))
        patches_nd.append(Rectangle((x, y), width, height, color = color, alpha = .75, ec = None, zorder = 2)) #use alphas to fade colors where there was no radar data

    p = PatchCollection(patches_nd, transform = ccrs.PlateCarree(), match_original=True)
    ax.add_collection(p)

    #======================================================================================================================================
    #create patches where radar data was used
    for i in range(len(grid['index'])):
        x = lon_ll[i]
        y = lat_ll[i]
        width = lon_lr[i] - lon_ll[i]
        height = lat_ul[i] - lat_ll[i]

        color = colormap(norm(vel_grid[i]))
        patches.append(Rectangle((x, y), width, height, color = color, alpha = None, ec = None, zorder = 3))

    p = PatchCollection(patches, transform = ccrs.PlateCarree(), match_original=True)
    ax.add_collection(p)

    #======================================================================================================================================
    #add and resize colorbar
    cbar_ax = fig.add_axes([0, 0, .1, .1]) #left, bottom, width, and height of axis; maximum of 1 for each value
    cbar_ax.set_position([0.9, .01, .02, .3]) #left, bottom, width, and height; set position of colorbar

    cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = 'plasma'), cax = cbar_ax, shrink = 0.5)
    cbar.ax.set_ylabel('Velocity Magnitude (m/s)')

    #======================================================================================================================================
    #plot meridians if flag is raised
    if meridians != None:
        for meridian in meridians:

            print("plotting meridian:",meridian)
            meridlon = aacgm.inv_mlt_convert(dfVel.dstr.yr, dfVel.dstr.mo, dfVel.dstr.dy, int(dfVel.dstr.hr), int(dfVel.dstr.mt), 0, meridian)
        
            ax.plot([meridlon, meridlon], [min_lat+2, max_lat-2], transform = ccrs.PlateCarree(), color = 'k', lw = 1.5, zorder = 2.5)

            label=cnv_mlt_label(meridian)
            ax.annotate(label, [meridlon + lon_delta, lab_lat], xycoords = transform, xytext = (meridlon + lon_delta, lab_lat),c = col, clip_on = True) 

    #======================================================================================================================================
    #title and save figure
    name_time_str = rtime.strftime("%H%M")
    name = make_date_str(dfVel.dstr) + '_' + name_time_str + '_velocity.png'
    plt.savefig(name)
    print('Plot has been made\n')
    plt.close()

    #======================================================================================================================================
    #advance dfVel file to next time 
    dfVel.read_record()
    if not isinstance(dfVel.dstr,read_df_record.record):
        break

#==========================================================================================================================================
#==========================================================================================================================================