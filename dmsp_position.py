#import required modules
import cdflib
import pandas as pd

from collections import defaultdict

#%%============================================================================
#define function to get magnetic latiude and longitude positions of each satellite
#==============================================================================
sats = ['f16', 'f17', 'f18']
sat_positions = defaultdict(list)

def sat_pos(sttime, ndtime):
    for sat in sats:
        try:
            #set path to cdf file
            date = str(sttime)[0:8]
            #data_path = '/import/SUPERDARN/matthew/dmsp/ssj4/' + date
            #sat = 'f17' #choose between f16, f17, and f18
    
            #cdf_file = cdflib.CDF(data_path + '/dmsp-' + sat + '_ssj_precipitating-electrons-ions_' + date + '_v1.1.2.cdf') #path on wilcox.met
            #cdf_file = cdflib.CDF('E:\Python Scripts\Research\dmsp-f17_ssj_precipitating-electrons-ions_20140321_v1.1.2.cdf') #path on Matthew-PC

            data_path = 'C:/Users\mflyn\Documents\Python Scripts\Research\dmsp/' + sat
            cdf_file = cdflib.CDF(data_path + '/dmsp-' + sat + '_ssj_precipitating-electrons-ions_' + date + '_v1.1.2.cdf') #path on matthew laptop
    
            #get variables from cdf
            epoch_data = cdf_file.varget('Epoch') #epoch time; this is the x-axis
            datetime_array = cdflib.cdfepoch.to_datetime(epoch_data) #convert CDF_EPOCH to datetime
            datetime_series = pd.Series(datetime_array) #convert datetime array to pandas series
    
            aacgm_lat = cdf_file.varget('SC_AACGM_LAT') #geomagnetic latitude
            aacgm_lon = cdf_file.varget('SC_AACGM_LON') #geomagnetic longitude
    
            #set time interval based on start and end times           
            sttime_dt = pd.to_datetime(sttime, format='%Y%m%d%H%M') #convert start time string to datetime object
            ndtime_dt = pd.to_datetime(ndtime, format='%Y%m%d%H%M') #convert end time string to datetime object

            interval = (datetime_series >= sttime_dt) & (datetime_series <= ndtime_dt) #set indicies that fall within the defined time interval
    
            aacgm_lat = aacgm_lat[interval]
            aacgm_lon = aacgm_lon[interval]
            datetime_series = datetime_series[interval]
            datetime_series = datetime_series.reset_index(drop = True)
            
            sat_positions[sat].append(aacgm_lat)
            sat_positions[sat].append(aacgm_lon)
            sat_positions[sat].append(datetime_series)
            
        except:
            continue
    
    return sat_positions



















