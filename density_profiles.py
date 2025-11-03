#import required modules
import pandas as pd
import matplotlib.pyplot as plt

#%%set path for file locations
path = 'C:/Users/User/OneDrive - The Pennsylvania State University/@GraduateSchool/Research/Masters Thesis/'

#set file paths
day_file = path + 'dayside_profiles.txt'
night_file = path + 'nightside_profiles.txt'

#read the files
df_day = pd.read_csv(day_file,
                     sep = '\s+', #columns separated by spaces
                     skiprows = 0, #skip the header text lines if needed
                     index_col = False #no columns in the .txt are indicies
                     )

df_night = pd.read_csv(night_file,
                     sep = '\s+', #columns separated by spaces
                     skiprows = 0, #skip the header text lines if needed
                     index_col = False #no columns in the .txt are indicies
                     )

#%%get date for each profile
hour = int(df_day['hour'][0])
day = int(df_day['Day'][0])
month = int(df_day['Mon'][0])
year = int(df_day['Year'][0])

#get dayside latiude and longitude
lat_day = int(df_day['Lat'][0])
lon_day = int(df_day['Lon'][0])

#plot dayside neutral density profile
fig = plt.figure(figsize = (4, 8))

x = df_day['air(gm/cm3)']
y = df_day['Heit(km)']

plt.plot(x, y)

plt.xscale('log')
plt.xlabel('Mass Density $(g/cm^3)$')
plt.ylabel('Altitude (km)')

#plt.title(f'{month}/{day}/{year} {str(hour)}00 UTC\nDayside ({lat_day}\u00B0N, {lon_day}\u00B0E) Neutral Density Profile')
plt.savefig('dayside_profile.png')

#get nightside latitude and longitude
lat_night = int(df_night['Lat'][0])
lon_night = int(df_night['Lon'][0])

#plot nightside neutral density profile
fig = plt.figure(figsize = (4, 8))

x = df_night['air(gm/cm3)']
y = df_night['Heit(km)']

plt.plot(x, y)

plt.xscale('log')
plt.xlabel('Mass Density $(g/cm^3)$')
plt.ylabel('Altitude (km)')

#plt.title(f'{month}/{day}/{year} {str(hour)}00 UTC\nNightside ({lat_night}\u00B0N, {lon_night}\u00B0E) Neutral Density Profile')
plt.savefig('nightside_profile.png')