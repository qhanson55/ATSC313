import math
import matplotlib.pyplot as plt
import numpy as np

countries = ['Brazil',
    'Canada',
    'China', 
    'DR Cong', 
    'Germany', 
    'India', 
    'Nepal', 
    'Norway', 
    'Pakistan', 
    'USA'
]
country_project_capacity = np.array([ # MW
    3750,
    1903,
    1320,
    1775,
    100,
    1500,
    144,
    116,
    1450,
    2620
])
country_seasonal_generation_rate = np.array([ # percentage of power 
    [1, 1, 0.25, 0.25],
    [0.40, 0.10, 1, 0.85], 
    [1, 1, 0.25, 0.25], 
    [1, 1, 0.25, 0.25],
    [0.40, 0.10, 1, 0.85],
    [0.40, 0.10, 1, 0.85],
    [1, 1, 0.25, 0.25],
    [1, 0.20, 0.60, 0.10], 
    [0.40, 0.10, 1, 0.85], 
    [0.40, 0.10, 1, 0.85]
])
country_electr_cost = np.array([ # [LLH (18hr), HLH (6hr)]
    [0.12, 0.25],
    [0.10, 0.21],
    [0.07, 0.18],
    [0.06, 0.06],
    [0.44, 0.44],
    [0.09, 0.09],
    [0.07, 0.11],
    [0.10, 0.10],
    [0.06, 0.11],
    [0.17, 0.17]
])
project_cost_rate = 3750 # $/kW


# calculate country/project's daily revenue based on LLH (18h) and HLH (6hr) rates
country_project_capacity_kW = [1000 * cpc for cpc in country_project_capacity]
rate_capacity_hour = (country_project_capacity_kW * country_electr_cost.T).T 
rate_capacity_day = [18, 6] * rate_capacity_hour
country_daily_generation = np.zeros(10)
for i in range(len(countries)):
    country_daily_generation[i] = rate_capacity_day[i][0] + rate_capacity_day[i][1]

# calculate longterm revenue with seasonal and yearly * 15 years
country_seasonal_generation = (country_daily_generation * country_seasonal_generation_rate.T).T
country_yearly_generation = np.sum(([90, 90, 90, 90] * country_seasonal_generation), axis = 1) 
country_longterm_generation = np.array([15 * cyg for cyg in country_yearly_generation])

# calculate project building costs and profit
country_project_cost = np.array([project_cost_rate * cpc for cpc in country_project_capacity_kW])
country_profit = country_longterm_generation - country_project_cost

print("Based on yearly revenue the countries rank in this order:")
for i in np.argsort(-country_yearly_generation):
    print(countries[i] + " - $" + str(country_yearly_generation[i]))
print()
print("Based on project costs rank in this order:")
for i in np.argsort(-country_project_cost):
    print(countries[i] + " - $" + str(country_project_cost[i]))
print()
print("Based on 15 year profit the countries rank in this order:")
for i in np.argsort(-country_profit):
    print(countries[i] + " - $" + str(country_profit[i]))