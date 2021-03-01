import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

wind_data = pd.read_csv('A313-wind-power-data.csv')

wind_speeds = wind_data['Wind Speed (m/s)']

# part a
wind_speed_counts = plt.hist(wind_speeds, bins=range(24), rwidth=0.95)
plt.xticks(range(0, 25))
plt.ylabel('Count')
plt.xlabel('Wind Speed [m/s]')
plt.title('Chetwynd Wind Speed Frequency for One Year')
plt.xlim((0,24))
# plt.show()

# part b
wind_speed_rel_feq = np.array(wind_speed_counts[0]) / len(wind_speeds)
plt.plot(wind_speed_rel_feq, 'b.')
plt.xticks([0,5,10,15,20,25])
plt.ylabel('Relative Frequency')
plt.xlabel('Wind Speed [m/s]')
plt.title('Chetwynd Relative Wind Speed Frequency for One Year')
# plt.show()

# part c
wind_speed_bins = np.array(wind_speed_counts[0])
wind_speeds_mean = np.mean(np.array(wind_speeds))
wind_speeds_median = np.median(np.array(wind_speeds))
wind_speeds_var = np.var(np.array(wind_speeds))
print(wind_speeds_mean)
print(wind_speeds_median)
print(wind_speeds_var)
# plt.plot(wind_speed_rel_feq, )
