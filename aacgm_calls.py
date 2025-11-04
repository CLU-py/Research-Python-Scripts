#!/usr/bin/env python3
#import required moduels

import aacgm
from datetime import datetime

#MLT = input('enter MLT value: ')
MLT = 21
date = datetime.now()
date = date.replace(year = 2024)

aacgm.set_datetime(date.year, date.month, date.day, date.hour, date.minute, 0) #sets the date for the IMF for the proper coordinates
conv_lon = aacgm.inv_mlt_convert(date.year, date.month, date.day, date.hour, date.minute, 0, MLT) #converts a given MLT to longitude
conv_MLT = aacgm.mlt_convert(date.year, date.month, date.day, date.hour, date.minute, 0, 310) #converts a given magnetic longitude to MLT

print(conv_lon)
print(conv_MLT)
