#!/usr/bin/env python3
#THIS SCRIPT MAKES A GRID FOR PLOTTING ON HEMISPHERICAL MAPS
#TO CALL THIS SCRIPT: from map_grid import make_grid 
#IF SCRIPT IS NOT EXECUTABLE IN TERNIMAL, RUN chmod +x {filename}.py

#import required modules
import numpy as np

def make_grid(lat_min = 55, dlat = 1):
    dtor = np.pi/180

    nlats = int((90-lat_min)/dlat) #number of latitude points
    #dlon = dlat/np.cos(lat_min * dtor) #change in longitude

    lats = []
    lons = []

    index = 0
    lat_index = 0
    lon_index = 0
    cells = {'index':[], 'lat_index':[], 'lon_index':[], 'lat_ll':[], 'lon_ll':[],
             'lat_lr':[], 'lon_lr':[], 'lat_ul':[], 'lon_ul':[], 'lat_ur':[], 'lon_ur':[]}

    for jt in range(nlats):
        lat_l = lat_min + jt * dlat
        lat_u = lat_min + (jt+1) * dlat
        lat = (lat_l + lat_u)/2

        dlon0 = dlat/np.cos(lat * dtor) #initial change in longitude at lowest latitude
        nlons = int(np.round(360/dlon0))
        dlon = 360/nlons
        lon0 = (dlon - dlon0)/2
        #lon_index = 0

        for jn in range(nlons):
            lon = lon0 + jn * dlon
            cells["index"].append(index)
            cells["lat_index"].append(lat_index)
            cells["lon_index"].append(lon_index)

            cells["lat_ll"].append(lat_l) #latitude in lower left of grid
            cells["lon_ll"].append(lon) #longitude in lower left of grid
            
            cells["lat_lr"].append(lat_l) #latitude in lower right of grid
            cells["lon_lr"].append(lon + dlon) #longitude in lower right of grid
            
            cells["lat_ul"].append(lat_u) #latitude in upper left of grid
            cells["lon_ul"].append(lon) #longitude in upper left of grid
            
            cells["lat_ur"].append(lat_u) #latitude in upper right of grid
            cells["lon_ur"].append(lon + dlon) #longitude in upper right of grid

            index += 1 #step foward in indicies
            lon_index += 1
            lat_index += 1

    #make the cells numpy arrays
    cells["index"] = np.asarray(cells["index"])
    cells["lat_index"] = np.asarray(cells["lat_index"])
    cells["lon_index"] = np.asarray(cells["lon_index"])
    
    cells["lat_ll"] = np.asarray(cells["lat_ll"])
    cells["lat_lr"] = np.asarray(cells["lat_lr"])

    cells["lat_ul"] = np.asarray(cells["lat_ul"])
    cells["lat_ur"] = np.asarray(cells["lat_ur"])

    cells["lon_ll"] = np.asarray(cells["lon_ll"])
    cells["lon_lr"] = np.asarray(cells["lon_lr"])
    
    cells["lon_ul"] = np.asarray(cells["lon_ul"])
    cells["lon_ur"] = np.asarray(cells["lon_ur"])

    return(cells)

#grid = make_grid()
#print('Map grid index:', grid['index'])
#print('Map grid latitude index:', grid['lat_index'])
#print('Map grid longitude index:', grid['lon_index'])

#print('\nMap grid upper left latitude:', grid['lat_ul'])
#print('Map grid upper left longitude:', grid['lon_ul'])

#print('\nMap grid upper right latitude:', grid['lat_ur'])
#print('Map grid upper right longitude:', grid['lon_ur'])

#print('\nMap grid lower left latitude:', grid['lat_ll'])
#print('Map grid lower left longitude:', grid['lon_ll'])

#print('\nMap grid lower right latitude:', grid['lat_lr'])
#print('Map grid lower right longitude:', grid['lon_lr'])

#mark start and end indices of each line of latitude
grid = make_grid()
def LatRange():
    lats_start = np.array([grid['lat_index'][0], grid['lat_index'][204], grid['lat_index'][403], grid['lat_index'][596], grid['lat_index'][784], grid['lat_index'][967],
                           grid['lat_index'][1144], grid['lat_index'][1316], grid['lat_index'][1482], grid['lat_index'][1643], grid['lat_index'][1798], grid['lat_index'][1947],
                           grid['lat_index'][2091], grid['lat_index'][2229], grid['lat_index'][2361], grid['lat_index'][2487], grid['lat_index'][2607], grid['lat_index'][2721],
                           grid['lat_index'][2829], grid['lat_index'][2931], grid['lat_index'][3030], grid['lat_index'][3117], grid['lat_index'][3201], grid['lat_index'][3279],
                           grid['lat_index'][3351], grid['lat_index'][3417], grid['lat_index'][3476], grid['lat_index'][3529], grid['lat_index'][3576], grid['lat_index'][3617],
                           grid['lat_index'][3652], grid['lat_index'][3680], grid['lat_index'][3702], grid['lat_index'][3718], grid['lat_index'][3727]]) #index of the start of each line of latitude
                           
    lats_end = np.array([grid['lat_index'][203], grid['lat_index'][402], grid['lat_index'][595], grid['lat_index'][783], grid['lat_index'][966], grid['lat_index'][1143],
                         grid['lat_index'][1315], grid['lat_index'][1481], grid['lat_index'][1642], grid['lat_index'][1797], grid['lat_index'][1946], grid['lat_index'][2090],
                         grid['lat_index'][2228], grid['lat_index'][2360], grid['lat_index'][2486], grid['lat_index'][2606], grid['lat_index'][2720], grid['lat_index'][2828],
                         grid['lat_index'][2930], grid['lat_index'][3029], grid['lat_index'][3116], grid['lat_index'][3200], grid['lat_index'][3278], grid['lat_index'][3350],
                         grid['lat_index'][3416], grid['lat_index'][3475], grid['lat_index'][3528], grid['lat_index'][3575], grid['lat_index'][3616], grid['lat_index'][3651],
                         grid['lat_index'][3679], grid['lat_index'][3701], grid['lat_index'][3717], grid['lat_index'][3726], grid['lat_index'][3729]]) #index of the end of each line of latitude
 
    return(lats_start, lats_end)