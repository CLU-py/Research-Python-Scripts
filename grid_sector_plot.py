#!/usr/bin/env python3
#import required modules
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import cartopy .crs as ccrs
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

from datetime import datetime
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

data = []

#%%add commands to call in terminal
parser = argparseparser = argparse.ArgumentParser(description='sector_gdf_vel argument parser')
parser.add_argument('date', type = str) #event date
parser.add_argument('--vectors', action = 'store_true') #command to generate velocity plot using vectors instead of magnitude

args = parser.parse_args()
vectors = args.vectors
event_date = args.date

#read text file into script
with open(f'/import/SUPERDARN/matthew/{event_date}/data_text_new/{event_date}_text.txt') as file: #path for file on wilcox.met
#with open('20140321_text.txt') as file: #path for file on SSH
    print('opened file')
    headers = file.readline().strip().split(', ') #get headers from first line
    
    file.readline() #read second line
    substorm_location = file.readline().strip() #read the third line to get substorm location
    
    for line in file:
        line = line.strip()
        if line == '':
            continue
        
        #get substorm onset latitude and longitude
        
        #find if the line is a timestamp
        if line[0].isdigit() and ':' in line:
            current_time = line
            continue
        
        #split and append the data
        parts = line.split()
        if len(parts) == 9:
            center_lat, center_lon = parts[0], parts[1]
            gridx, gridy = int(parts[2]), int(parts[3])
            velocity = parts[4]
            velx = parts[5]
            vely = parts[6]
            brightness = parts[7]
            covariance = parts[8]
            
            data.append([current_time, center_lat, center_lon, gridx, gridy, velocity, velx, vely, brightness, covariance])
            
    #create dataframe
    df = pd.DataFrame(data, columns = ['time','center lat', 'center lon', 'gridx', 'gridy', 'velocity', 'velx', 'vely', 'brightness', 'covariance'])
   

#substorm_lat, substorm_lon = map(int, substorm_location.split()) #set substorm lat/lon to separate variables
substorm_lat, substorm_lon, substorm_MLT, region_centerlon = map(float, substorm_location.split()) #set substorm lat/lon to separate variables
lat_diff = substorm_lat - 65 #degrees latitude substorm latitude is from 65
lon_diff = substorm_lon - 360 #degrees longitude substorm longitude is from 0

#%%get latitude and longitude corners for grid cells to plot
dtor = np.pi/180
dlat = 1

#add plot parameters
theta = np.linspace(0, 2*np.pi, 360)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T

circle = mpath.Path(verts * radius + center)

#%%loop through times
for time in df['time'].unique():
    collection = [] #empty list to hold the three patch collections
    print(f'\n{time}')
    
    data_set = df[df['time'] == time] #create a data set for values for the current time
    
    center_lats = data_set['center lat'].values
    center_lons = data_set['center lon'].values
    
    velocity = data_set['velocity'].to_numpy().astype(float)
    velx = data_set['velx'].to_numpy().astype(float)
    vely = data_set['vely'].to_numpy().astype(float)
    brightness = data_set['brightness'].to_numpy().astype(float)
    covariance = data_set['covariance'].to_numpy().astype(float)
    
    center_lats = np.array(center_lats, dtype = float) #array of the center latitudes for the current data set
    center_lons = np.array(center_lons, dtype = float) #array of the center longitudes for the current data set
    
    center_lats_adj = center_lats[:] - lat_diff #adjust the latitude value based on the latitude difference from 65, USE THIS TO GET DLAT
    lat_ul = center_lats_adj[:] + 0.5 #upper left latitude of grid cells
    lat_ll = center_lats_adj[:] - 0.5 #lower left latitude of grid cells
    
    center_lons_adj = center_lons[:] - lon_diff #adjust the lontitude value based on the longitude difference from zero
    center_lons_adj = np.where(center_lons_adj > 360, center_lons_adj - 360, center_lons_adj)
    
    #%%loop through latitudes to get dlon for each latitude
    center_lats_unique = np.unique(center_lats_adj) #list of unique latitudes
    
    lon_lr = np.zeros(len(center_lats))
    lon_ll = np.zeros(len(center_lats))
    
    for lat in center_lats_unique:
        lat_indicies = np.where(center_lats_adj == lat) #indicies for each line of latitude
        
        dlon0 = dlat/np.cos(lat * dtor)
        nlons = int(np.round(360/dlon0))
        dlon = 360/nlons
        
        for index in lat_indicies[0]:
            lon_lr[index] = center_lons_adj[index] + (dlon/2) #lower right longitude of grid cells
            lon_ll[index] = center_lons_adj[index] - (dlon/2) #lower right lontitude of grid cells
            
    grid_indx = data_set.index[data_set['gridx'] == 0].tolist() #list of indices for the first row of the grid
    
    lon_values = data_set.loc[grid_indx, ['center lon']] #longitude values from the first row on the grid
    lon_values['center lon'] = lon_values['center lon'].astype(float) #convert strings to floats
    lon_values['center lon'] = lon_values['center lon'].apply(lambda x: x + 360 if x < 180 else x) #if values are less than 180, add 360 to them
    first_lon = lon_values['center lon'].min() #get the first longitude point
    last_lon = lon_values['center lon'].max() #get the last longitude point
    center_lon = (first_lon + last_lon) / 2 #find the center point between the first and last points
    center_lon = center_lon - lon_diff #adjust central longitude based on longitude difference from 0
    #print(center_lon)
            
    #%%create figure and coordinate reference system to plot the data
    min_lat = 55
    max_lat = 90
    min_lon = 0
    max_lon = 360
    
    lat_north = 65 + 5
    lat_south = 65 - 5
    lon_east = float(last_lon) + 10
    #if lon_east > 360:
        #lon_east = lon_east - 360
        
    lon_west = float(first_lon) - 10
    if lon_west > 360:
        lon_west = lon_west - 360
        
    lon_east = lon_east - lon_diff
    lon_west = lon_west - lon_diff
        
    #print(lon_west, lon_east)
        
    latitude_lines = [55, 60, 65, 70, 75, 80, 85, 90] #lines of latitude to include in the background grid
    latitudes = np.linspace(lat_south - 1, lat_north + 2, 100) #list of latitude points to plot a longitude line along
    longitudes = np.linspace(lon_west, lon_east, 100) #list of longitude points to plot a longitude line along
    
    fig, axes = plt.subplots(3, 1, figsize = (10, 15), dpi = 200, subplot_kw = {'projection': ccrs.AzimuthalEquidistant(
        central_longitude = center_lon, central_latitude = 65, false_easting = 0.0, false_northing = 0.0, globe = None)})
    
    for ax in axes:
        ax.set_adjustable('datalim')
        ax.set_extent((lon_west, lon_east, lat_south, lat_north), ccrs.PlateCarree())  #set projection limits
        #ax.set_extent((min_lon, max_lon, min_lat, max_lat), ccrs.PlateCarree())  #set projection limits
        #ax.set_boundary(circle, transform=ax.transAxes)  #set circular boundary
        gl = ax.gridlines()  #add latitude/longitude grid lines
        gl.ylocator = plt.FixedLocator(latitude_lines) #specify specific lines of latitude to plot
        ax.set_clip_on(True)
        
        ax.set_facecolor('grey')
        
        #plot lines of latitude and longitude aligned with substorm onset
        ax.plot(longitudes, [65] * len(longitudes),
                transform = ccrs.PlateCarree(), color = 'black', linewidth = 5, linestyle = '--')
        
        ax.plot([substorm_lon - lon_diff] * len(latitudes), latitudes,
                transform = ccrs.PlateCarree(), color = 'black', linewidth = 5, linestyle = '--')
    
        #add annotation for time and line lables
        xpos, ypos = 0.02, 0.93
        ax.annotate(f'{time}', (xpos, ypos), xycoords = 'axes fraction', fontsize = 16,
                    bbox = dict(facecolor = 'white', edgecolor = 'none', boxstyle = 'round, pad = 0.3'))
        
        #xpos, ypos = 0.72, 0.05
        #ax.annotate(f'Substorm Onset\nMLT ({substorm_MLT})', (xpos, ypos), xycoords = 'axes fraction', fontsize = 11,
                    #bbox = dict(facecolor = 'white', edgecolor = 'none', boxstyle = 'round, pad = 0.3'), ha = 'center')
        
        ax.annotate(f'Substorm Onset\nMLT ({substorm_MLT})',
                    xy = (substorm_lon - lon_diff, substorm_lat + 6),
                    xycoords = ccrs.PlateCarree()._as_mpl_transform(ax),
                    fontsize = 11,
                    bbox = dict(facecolor = 'white', edgecolor = 'none', boxstyle = 'round, pad = 0.3'),
                    ha = 'center'
                    )
        
        xpos, ypos = 0.09, 0.75
        ax.annotate(f'Substorm Onset\nLatitude ({substorm_lat})', (xpos, ypos), xycoords = 'axes fraction', fontsize = 11,
                    bbox = dict(facecolor = 'white', edgecolor = 'none', boxstyle = 'round, pad = 0.3'), ha = 'center')
        
    #%%create patches for velocity
    if vectors:
        print('creating velocity plot using vectors')

        #get speed for colormap
        speed = np.sqrt(velx**2 + vely**2)
        vel_norm = Normalize(vmin = 0, vmax = 800)

        #create quiver plot on first subplot axis
        q = axes[0].quiver(
            (lon_ll + lon_lr)/2, (lat_ll + lat_ul)/2,
            velx, vely,
            speed,
            cmap = 'gnuplot',
            norm = vel_norm,
            scale = 1000,
            scale_units='inches',
            width = 0.003,
            transform = ccrs.PlateCarree()
        )

        #add velocity colorbar
        cbar_ax = fig.add_axes([0.88, 0.86, .02, .12])
        cbar = fig.colorbar(cm.ScalarMappable(norm = vel_norm, cmap = 'gnuplot'), cax = cbar_ax, shrink = 0.5)
        cbar.ax.set_ylabel('Plasma Velocity (m/s)')

    else:
        print('creating velocity plot using magnitude')
        vel_colormap = mpl.colormaps['gnuplot'] #get colormap object from matplotlib colormap
        vel_norm = Normalize(vmax = 800, vmin = 0)
    
        vel_patches = []
    
        for i in range(len(center_lats)):
            x = lon_ll[i]
            y =lat_ll[i]
            width = lon_lr[i] - lon_ll[i]
            height = lat_ul[i] - lat_ll[i]
        
            color = vel_colormap(vel_norm(velocity[i]))
        
            vel_patches.append(Rectangle((x, y), width, height, color = color))
    
        velP = PatchCollection(vel_patches, transform = ccrs.PlateCarree(), match_original = True)
        collection.append(velP)
    
        #add colorbar for velocity
        cbar_ax = fig.add_axes([0, 0, .1, .1]) #create new axis for colorbar; left, bottom, width, and height of axis (maximum of 1 for each value)
        cbar_ax.set_position([0.88, 0.86, .02, .12]) #eft, bottom, width, and height; set position of colorbar
    
        cbar = fig.colorbar(cm.ScalarMappable(norm = vel_norm, cmap = vel_colormap), cax = cbar_ax, shrink = 0.5)
        cbar.ax.set_ylabel('Plasma Velocity (m/s)')
    
    #%%create patches for brightness
    bright_colormap = mpl.colormaps['rainbow'] #get colormap object from matplotlib colormap
    grey = (100/255, 100/255, 100/255)
    bright_norm = Normalize(vmax = 5000, vmin = 0)
    
    bright_patches = []
    
    for i in range(len(center_lats)):
        x = lon_ll[i]
        y = lat_ll[i]
        width = lon_lr[i] - lon_ll[i]
        height = lat_ul[i] - lat_ll[i]

        if brightness[i] == 0:
            bright_patches.append(Rectangle((x, y), width, height, color = grey))

        else:
            color = bright_colormap(bright_norm(brightness[i]))
            bright_patches.append(Rectangle((x, y), width, height, color = color))
    
    brightP = PatchCollection(bright_patches, transform = ccrs.PlateCarree(), match_original = True)
    collection.append(brightP)
    
    #add colorbar for brightness
    cbar_ax = fig.add_axes([0, 0, .1, .1]) #create new axis for colorbar; left, bottom, width, and height of axis (maximum of 1 for each value)
    cbar_ax.set_position([0.88, 0.53, .02, .12]) #eft, bottom, width, and height; set position of colorbar
    
    cbar = fig.colorbar(cm.ScalarMappable(norm = bright_norm, cmap = bright_colormap), cax = cbar_ax, shrink = 0.5)
    cbar.ax.set_ylabel('Auroral Brightness')
    
    #%%create patches for covariance
    cov_colormap = mpl.colormaps['RdYlGn'] #get colormap object from matplotlib colormap
    cov_norm = Normalize(vmax = 350000, vmin = -350000)
    
    cov_patches = []
    
    for i in range(len(center_lats)):
        x = lon_ll[i]
        y = lat_ll[i]
        width = lon_lr[i] - lon_ll[i]
        height = lat_ul[i] - lat_ll[i]

        if brightness[i] == 0:
            cov_patches.append(Rectangle((x, y), width, height, color = grey))

        else:
            color = cov_colormap(cov_norm(covariance[i]))
            cov_patches.append(Rectangle((x, y), width, height, color = color))
    
    covP = PatchCollection(cov_patches, transform = ccrs.PlateCarree(), match_original = True)
    collection.append(covP)
    
    #add colorbar for covariance
    cbar_ax = fig.add_axes([0, 0, .1, .1]) #create new axis for colorbar; left, bottom, width, and height of axis (maximum of 1 for each value)
    cbar_ax.set_position([0.88, 0.20, .02, .12]) #eft, bottom, width, and height; set position of colorbar
    
    cbar = fig.colorbar(cm.ScalarMappable(norm = cov_norm, cmap = cov_colormap), cax = cbar_ax, shrink = 0.5)
    cbar.ax.set_ylabel('Covariance (Unitless)')
    ticks = cbar.get_ticks()
    cbar.ax.set_yticklabels([f'{tick:.0e}' for tick in ticks])
    
    #%%plot patches and colorbars
    colormaps = ['plasma', 'rainbow', 'RdYlGn']
    data_types = ['Velocity (m/s)', 'Brightness', 'Covariance']
    #norm_ranges = [(0, 1500), (0, 5000), (-35000, 35000)]

    axes[1].add_collection(brightP)
    axes[2].add_collection(covP)

    if not vectors:
        axes[0].add_collection(velP)

    plt.tight_layout()

    #format time string for saving figure
    dt = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    time = dt.strftime('%Y%m%d_%H%M')
    name = f'{time}_sector.png'
    plt.savefig(f'/import/SUPERDARN/matthew/{event_date}/grid_sector_plots_new/{name}') #save figure
    #plt.savefig(f'/import/SUPERDARN/matthew/{event_date}/grid_sector_plots_ts18/{name}') #save figure
    plt.close()

print('plots made')
    
#annotate line along longitude and latitude circle

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
