#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:38:19 2021

@author: alejandro
"""

# =============================================================================
# Script to climatological studies 
# By Alejandro Rodríguez Sánchez
# =============================================================================

# Step 0: Import libraries and define functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib.dates import DateFormatter, MinuteLocator
import seaborn as sns
import matplotlib.cm as cm
import os
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from rpy2.robjects.packages import importr
from scipy import interpolate
from scipy.stats import gumbel_r
from pandas import Grouper,DataFrame
import pymannkendall as mk
import matplotlib as mpl



def convertir_a_calendarionatural(variable_diaria):
    variable_diaria_natural = variable_diaria
    variable_diaria_natural = np.insert(variable_diaria_natural,366,variable_diaria[0:122,:],axis=0)
    variable_diaria_natural = np.delete(variable_diaria_natural,slice(0,121),0)
    variable_diaria_natural = np.delete(variable_diaria_natural,0,0)
    
    variable2010 = variable_diaria[0:122,0]
    dias2010 = np.zeros(244,float)
    dias2010[:] = np.nan
    variable2010 = np.append(dias2010,variable2010,axis=0)
    variable2010 = variable2010[:,None]
    variable_con2010 = np.append(variable2010,variable_diaria_natural,axis=1)
    for i in range(1,nyears):
        variable_con2010[244:366,i] = variable_con2010[244:366,i+1]
    
    variable_diaria_natural = variable_con2010
    variable_diaria_natural = np.delete(variable_diaria_natural,nyears,axis=1)

    
    return variable_diaria_natural

def linear_regression(x,y,alpha=0.05):
    # Determino la recta de regresión
    mean_y = y.mean()
    mean_x = x.mean()
    cov_xy = ((x*y).sum() - len(x)*mean_x*mean_y)/(len(x)-1)
    var_x = ((x**2).sum() - len(x)*mean_x**2)/(len(x)-1)
    var_y = ((y**2).sum() - len(y)*mean_y**2)/(len(y)-1)
    y_reg = mean_y + (cov_xy/var_x)*(x-mean_x)
    y_reg_slope = (y_reg[-1] - y_reg[0])/len(x)

    # Determino el intervalo de confianza de la recta de regresión
    b = cov_xy/var_x
    from scipy import stats
    t_alfamedios = stats.t.ppf(1-(0.5*alpha),len(x)-2)
    sr = np.sqrt(((y-y_reg)**2).sum()/(len(x)-2))
    termino_raiz = np.sqrt(1+(1/len(x))+((x-mean_x)**2/((len(x)-1)*var_x)))
    high_confidence = y_reg + t_alfamedios*sr*termino_raiz
    low_confidence = y_reg - t_alfamedios*sr*termino_raiz

    return y_reg, y_reg_slope, high_confidence, low_confidence, alpha


##### Step 1: Import databases.
input_file_wind = "SerieVientoCampillos.txt"
input_file_winddir = "SerieDirViento_Campillos.txt"
input_file_T = "TemperaturasCampillos.txt"
input_file_prec = "PrecipitacionesCampillos.txt"

# Location
location = str(input('Input the location where measures have been done: '))

# SET UNITS
windunits = str(input('Input wind units (e.g.: km/h): '))
tempunits = str(input('Input temperature units (e.g.: ºC): '))
precunits = str(input('Input precipitation units (e.g.: mm): '))

dfwind = pd.read_csv(input_file_wind,sep = "\t", decimal = ",", names = ["MeanVelocity [%s]" %windunits,"MaxVelocity [%s]" %windunits])
dfwind[dfwind==-999]=np.nan
dfwind = dfwind.astype(float)

dfwinddir = pd.read_csv(input_file_winddir,sep = "\t", names = ["WindDirection [º]"])
dfwinddir[dfwinddir==-999]=np.nan

dfT = pd.read_csv(input_file_T,sep = "\t", decimal = ".", names = ["MinTemp [%s]" %tempunits,"MaxTemp [%s]" %tempunits,"MeanTemp [%s]" %tempunits])
dfT[dfT==-999]=np.nan
dfT = dfT.astype(float)

dfprec = pd.read_csv(input_file_prec,sep = "\t", decimal = ".", names = ["DailyPrec [%s]" %precunits])
dfprec[dfprec==-999]=np.nan
dfprec = dfprec.astype(float)

# Convert wind direction data to numerical data if input data is not numerical 
if dfwinddir['WindDirection [º]'].dtypes != 'float64':
    for i in range(len(dfwinddir)):
        if dfwinddir['WindDirection [º]'][i] in ['N']:
            dfwinddir['WindDirection [º]'][i]=0
        elif dfwinddir['WindDirection [º]'][i] in ['NNE']:
            dfwinddir['WindDirection [º]'][i]=22.5
        elif dfwinddir['WindDirection [º]'][i] in ['NE']:
            dfwinddir['WindDirection [º]'][i]=45
        elif dfwinddir['WindDirection [º]'][i] in ['ENE']:
            dfwinddir['WindDirection [º]'][i]=67.5
        elif dfwinddir['WindDirection [º]'][i] in ['E']:
            dfwinddir['WindDirection [º]'][i]=90
        elif dfwinddir['WindDirection [º]'][i] in ['ESE']:
            dfwinddir['WindDirection [º]'][i]=112.5
        elif dfwinddir['WindDirection [º]'][i] in ['SE']:
            dfwinddir['WindDirection [º]'][i]=135
        elif dfwinddir['WindDirection [º]'][i] in ['SSE']:
            dfwinddir['WindDirection [º]'][i]=157.5
        elif dfwinddir['WindDirection [º]'][i] in ['S']:
            dfwinddir['WindDirection [º]'][i]=180
        elif dfwinddir['WindDirection [º]'][i] in ['SSW']:
            dfwinddir['WindDirection [º]'][i]=202.5
        elif dfwinddir['WindDirection [º]'][i] in ['SW']:
            dfwinddir['WindDirection [º]'][i]=225
        elif dfwinddir['WindDirection [º]'][i] in ['WSW']:
            dfwinddir['WindDirection [º]'][i]=247.5
        elif dfwinddir['WindDirection [º]'][i] in ['W']:
            dfwinddir['WindDirection [º]'][i]=270
        elif dfwinddir['WindDirection [º]'][i] in ['WNW']:
            dfwinddir['WindDirection [º]'][i]=292.5
        elif dfwinddir['WindDirection [º]'][i] in ['NW']:
            dfwinddir['WindDirection [º]'][i]=315
        elif dfwinddir['WindDirection [º]'][i] in ['NNW']:
            dfwinddir['WindDirection [º]'][i]=337.5
        
dfwinddir[dfwinddir==-999]=np.nan
dfwinddir['WindDirection [º]'] = dfwinddir['WindDirection [º]'].astype(float)

# Merge all data in one dataframe and set Datetime index
climatedata_df = pd.concat([dfT, dfprec, dfwind, dfwinddir],axis=1)
climatedata_df_copy = climatedata_df

# Add data to both dataframes
date1 = '2010-09-01' # Date of first data
date2 = '2021-06-10' # Date of last data
datetoday = date2
datehoy = date2
mydates = pd.date_range(date1, date2).tolist()
climatedata_df['Date'] = mydates
climatedata_df.set_index('Date',inplace=True)  # Para que los índices sean fechas y así se ponen en el eje x de forma predeterminada
climatedata_df.index = climatedata_df.index.to_pydatetime()
climatedata_df['Day'] = climatedata_df.index.day
climatedata_df['Month'] = climatedata_df.index.month
climatedata_df['Year'] = climatedata_df.index.year
dias_index = climatedata_df.index.day
meses_index = climatedata_df.index.month
years_index = climatedata_df.index.year

monthtoday = int(climatedata_df.index.month[-1])
yearinicio = int(climatedata_df.index.year[0])
yeartoday= int(climatedata_df.index.year[-1])
nyears = int(yeartoday-yearinicio)+1

julian_days = [0,31,60,91,121,152,182,213,244,274,305,335,366]
daytoday = julian_days[monthtoday-1] + int(climatedata_df.index.day[-1])-1
# Add missing data if initial and/or final year are not completed
# Leap years
leapyears=[2020,2024,2028,2032,2036,2040,2044,2048,2052,2056,2060]
if yeartoday not in leapyears:
    ndaysyear=365
else:
    ndaysyear=366
missing_data_initialyear = np.zeros([365-122,np.size(climatedata_df,1)],float)
missing_data_initialyear[:] = np.nan
missing_data_initialyear = pd.DataFrame(missing_data_initialyear)
missing_data_initialyear.columns = climatedata_df.columns
missing_data_currentyear = np.zeros([ndaysyear-daytoday,np.size(climatedata_df,1)],float)
missing_data_currentyear[:] = np.nan
missing_data_currentyear = pd.DataFrame(missing_data_currentyear)
missing_data_currentyear.columns = climatedata_df.columns

# climatedata_df_complete = DataFrame()
climatedata_df_complete = pd.concat([missing_data_initialyear, climatedata_df_copy, missing_data_currentyear], ignore_index=True,axis=0)



date0 = str(yearinicio)+'-01-01' # 1st day of the year of the first data
date31dec = str(yeartoday)+'-12-31' # last day of the year of the last data
mydates_complete = pd.date_range(date0, date31dec).tolist()
climatedata_df_complete['Date'] = mydates_complete
climatedata_df_complete.set_index('Date',inplace=True)
climatedata_df_complete.index = climatedata_df_complete.index.to_pydatetime()

years_array = np.arange(yearinicio,yeartoday+1,1)



##### Step 2: Reshape data to calculate percentiles, distribution functions, etc.
# def create_timeseries(climatedata_df):
    
#climatedata_reshaped = np.zeros([366,nyears+2,np.size(climatedata_df,1)-3])
climatedata_reshaped = np.zeros([np.size(climatedata_df,1)-3,366,nyears+2])
climatedata_reshaped[:] = np.nan
# Be aware of which matrix corresponds to each variable


for var in range(np.size(climatedata_df,1)-3):
    k=1
    l=0
    for h in range(0,365):
        climatedata_reshaped[var,h,0]=climatedata_df.iloc[h,var]
    for i in range(len(climatedata_df)):
        for j in range(0,365):
            if climatedata_df.Day[i]==climatedata_df.Day[j] and climatedata_df.Month[i]==climatedata_df.Month[j] and climatedata_df.Year[i]!=climatedata_df.Year[j]:
                climatedata_reshaped[var,l,k]=climatedata_df.iloc[i,var]
                climatedata_reshaped[var,l,-2]=climatedata_df.Day[i]
                climatedata_reshaped[var,l,-1]=climatedata_df.Month[i]
                l=l+1
            else:
                continue
        if l==365:
            k=k+1
            l=0
        # Insert 29th February data at the end of the table
        if climatedata_df.Day[i]==29 and climatedata_df.Month[i]==2:
                climatedata_reshaped[var,365,k]=climatedata_df.iloc[i,var]
                climatedata_reshaped[var,365,-2]=climatedata_df.Day[i]
                climatedata_reshaped[var,365,-1]=climatedata_df.Month[i]

        
for var in range(np.size(climatedata_df,1)-3):
    a = climatedata_reshaped[var,:,:]
    a = np.insert(a, 181, a[365,:], axis=0)    # Insert 29th February data in its position
    a = np.delete(a,366,0)                          # Remove duplicate 29th February data   
    climatedata_reshaped[var,:,:] = a


# Reorganize matrices rows if data starts not in January 1st
climatedata_reshaped_naturalyear = np.zeros([np.size(climatedata_df,1)-3,366,nyears+2])
climatedata_reshaped_naturalyear[:] = np.nan
if climatedata_df.Month[0] != 1 or climatedata_df.Day[0] != 1:
    for i in range(np.size(climatedata_reshaped,0)):
        climatedata_reshaped_naturalyear[i,:,:] = convertir_a_calendarionatural(climatedata_reshaped[i,:,:])


# Identify dates in which extreme values of variables where registered
minimum_values_dates = []
maximum_values_dates = []

for i in range(np.size(climatedata_df,1)-3):
    minimum_values_dates.append(climatedata_df.index[climatedata_df.iloc[:,i] == np.nanmin(climatedata_df.iloc[:,i])])
    maximum_values_dates.append(climatedata_df.index[climatedata_df.iloc[:,i] == np.nanmax(climatedata_df.iloc[:,i])])

    minimum_values_dates[i] = minimum_values_dates[i][0].strftime("%d-%m-%Y ") # If extreme value is not unique, set as extreme date the first occurrence
    maximum_values_dates[i] = maximum_values_dates[i][0].strftime("%d-%m-%Y ")

# Calculate moving averages timeseries
n1 = 31        # Puntos sobre los que promediar

K = (n1-1)//2
moving_variables = np.zeros([len(climatedata_df),np.size(climatedata_df,1)-3])

for j in range(np.size(climatedata_df,1)-3):
    for i in range(K,len(climatedata_df)-K):
        moving_variables[i,j]=(1/(2*K+1))*sum(climatedata_df.iloc[i-K:i+K,j])

#    movil_pcp[i] = (1/(2*K+1))*sum(df1.Pcp[i-K:i+K])

moving_variables[:K,:] = np.nan
moving_variables[-K:,:] = np.nan

moving_variables = pd.DataFrame(moving_variables,columns=climatedata_df.columns[:-3])
moving_variables.index = climatedata_df.index

# Identify dates in which extreme values of variables where registered
minimum_movingvalues_dates = []
maximum_movingvalues_dates = []

for i in range(np.size(climatedata_df,1)-3):
    minimum_movingvalues_dates.append(moving_variables.index[moving_variables.iloc[:,i] == np.nanmin(moving_variables.iloc[:,i])])
    maximum_movingvalues_dates.append(moving_variables.index[moving_variables.iloc[:,i] == np.nanmax(moving_variables.iloc[:,i])])

    minimum_movingvalues_dates[i] = minimum_movingvalues_dates[i][0].strftime("%d-%m-%Y ") # If extreme value is not unique, set as extreme date the first occurrence
    maximum_movingvalues_dates[i] = maximum_movingvalues_dates[i][0].strftime("%d-%m-%Y ")



# Calculate daily percentiles, means, medians and extreme values for each variable
p005values_dataframe = np.zeros([np.size(climatedata_df,1),366],float)
p010values_dataframe = np.zeros([np.size(climatedata_df,1),366],float)
p090values_dataframe = np.zeros([np.size(climatedata_df,1),366],float)
p095values_dataframe = np.zeros([np.size(climatedata_df,1),366],float)
medianvalues_dataframe = np.zeros([np.size(climatedata_df,1),366],float)
meanvalues_dataframe = np.zeros([np.size(climatedata_df,1),366],float)
minvalues_dataframe = np.zeros([np.size(climatedata_df,1),366],float)
maxvalues_dataframe = np.zeros([np.size(climatedata_df,1),366],float)

if climatedata_df.Month[0] != 1 or climatedata_df.Day[0] != 1:
    p005values_dataframe_naturalyear = np.zeros([np.size(climatedata_df,1),366],float)
    p010values_dataframe_naturalyear = np.zeros([np.size(climatedata_df,1),366],float)
    p090values_dataframe_naturalyear = np.zeros([np.size(climatedata_df,1),366],float)
    p095values_dataframe_naturalyear = np.zeros([np.size(climatedata_df,1),366],float)
    medianvalues_dataframe_naturalyear = np.zeros([np.size(climatedata_df,1),366],float)
    meanvalues_dataframe_naturalyear = np.zeros([np.size(climatedata_df,1),366],float)
    minvalues_dataframe_naturalyear = np.zeros([np.size(climatedata_df,1),366],float)
    maxvalues_dataframe_naturalyear = np.zeros([np.size(climatedata_df,1),366],float)

for j in range(np.size(climatedata_df,1)-3):
    for i in range(0,366):
        p005values_dataframe[j,i]=np.nanpercentile(climatedata_reshaped[j,i,0:nyears],5)
        p010values_dataframe[j,i]=np.nanpercentile(climatedata_reshaped[j,i,0:nyears],10)
        p090values_dataframe[j,i]=np.nanpercentile(climatedata_reshaped[j,i,0:nyears],90)
        p095values_dataframe[j,i]=np.nanpercentile(climatedata_reshaped[j,i,0:nyears],95)
        medianvalues_dataframe[j,i]=np.nanmedian(climatedata_reshaped[j,i,0:nyears])
        meanvalues_dataframe[j,i]=np.nanmean(climatedata_reshaped[j,i,0:nyears])
        minvalues_dataframe[j,i]=np.nanmin(climatedata_reshaped[j,i,0:nyears])
        maxvalues_dataframe[j,i]=np.nanmax(climatedata_reshaped[j,i,0:nyears])

        if climatedata_df.Month[0] != 1 or climatedata_df.Day[0] != 1:
            p005values_dataframe_naturalyear[j,i]=np.nanpercentile(climatedata_reshaped_naturalyear[j,i,0:nyears],5)
            p010values_dataframe_naturalyear[j,i]=np.nanpercentile(climatedata_reshaped_naturalyear[j,i,0:nyears],10)
            p090values_dataframe_naturalyear[j,i]=np.nanpercentile(climatedata_reshaped_naturalyear[j,i,0:nyears],90)
            p095values_dataframe_naturalyear[j,i]=np.nanpercentile(climatedata_reshaped_naturalyear[j,i,0:nyears],95)
            medianvalues_dataframe_naturalyear[j,i]=np.nanmedian(climatedata_reshaped_naturalyear[j,i,0:nyears])
            meanvalues_dataframe_naturalyear[j,i]=np.nanmean(climatedata_reshaped_naturalyear[j,i,0:nyears])
            minvalues_dataframe_naturalyear[j,i]=np.nanmin(climatedata_reshaped_naturalyear[j,i,0:nyears])
            maxvalues_dataframe_naturalyear[j,i]=np.nanmax(climatedata_reshaped_naturalyear[j,i,0:nyears])
        
# Project matrices into arrays for plotting timeseries
p005values_dataframe_timeserie = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
p010values_dataframe_timeserie = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
p090values_dataframe_timeserie = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
p095values_dataframe_timeserie = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
medianvalues_dataframe_timeserie = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
meanvalues_dataframe_timeserie = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
minvalues_dataframe_timeserie = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
maxvalues_dataframe_timeserie = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)

p005values_dataframe_timeserie_naturalyear = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
p010values_dataframe_timeserie_naturalyear = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
p090values_dataframe_timeserie_naturalyear = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
p095values_dataframe_timeserie_naturalyear = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
medianvalues_dataframe_timeserie_naturalyear = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
meanvalues_dataframe_timeserie_naturalyear = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
minvalues_dataframe_timeserie_naturalyear = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)
maxvalues_dataframe_timeserie_naturalyear = np.zeros([int(366*nyears),np.size(climatedata_df,1)-3],float)

for j in range(np.size(climatedata_df,1)-3):
    p005values_dataframe_timeserie[:,j] = np.tile(p005values_dataframe[j,:],nyears)
    p010values_dataframe_timeserie[:,j] = np.tile(p010values_dataframe[j,:],nyears)
    p090values_dataframe_timeserie[:,j] = np.tile(p090values_dataframe[j,:],nyears)
    p095values_dataframe_timeserie[:,j] = np.tile(p095values_dataframe[j,:],nyears)
    medianvalues_dataframe_timeserie[:,j] = np.tile(medianvalues_dataframe[j,:],nyears)
    meanvalues_dataframe_timeserie[:,j] = np.tile(meanvalues_dataframe[j,:],nyears)
    minvalues_dataframe_timeserie[:,j] = np.tile(minvalues_dataframe[j,:],nyears)
    maxvalues_dataframe_timeserie[:,j] = np.tile(maxvalues_dataframe[j,:],nyears)

    if climatedata_df.Month[0] != 1 or climatedata_df.Day[0] != 1:
        p005values_dataframe_timeserie_naturalyear[:,j] = np.tile(p005values_dataframe_naturalyear[j,:],nyears)
        p010values_dataframe_timeserie_naturalyear[:,j] = np.tile(p010values_dataframe_naturalyear[j,:],nyears)
        p090values_dataframe_timeserie_naturalyear[:,j] = np.tile(p090values_dataframe_naturalyear[j,:],nyears)
        p095values_dataframe_timeserie_naturalyear[:,j] = np.tile(p095values_dataframe_naturalyear[j,:],nyears)
        medianvalues_dataframe_timeserie_naturalyear[:,j] = np.tile(medianvalues_dataframe_naturalyear[j,:],nyears)
        meanvalues_dataframe_timeserie_naturalyear[:,j] = np.tile(meanvalues_dataframe_naturalyear[j,:],nyears)
        minvalues_dataframe_timeserie_naturalyear[:,j] = np.tile(minvalues_dataframe_naturalyear[j,:],nyears)
        maxvalues_dataframe_timeserie_naturalyear[:,j] = np.tile(maxvalues_dataframe_naturalyear[j,:],nyears)
        
# Remove extra leapyear-days on timeseries arrays
leapdays_indexes = np.arange(181,len(climatedata_df),366)     

if climatedata_df.Month[0] != 1 or climatedata_df.Day[0] != 1:
    natural_leapdays_indexes = np.arange(59,len(climatedata_df),366)     


p005values_dataframe_timeserie = np.delete(p005values_dataframe_timeserie,leapdays_indexes,axis=0)
p010values_dataframe_timeserie = np.delete(p010values_dataframe_timeserie,leapdays_indexes,axis=0)
p090values_dataframe_timeserie = np.delete(p090values_dataframe_timeserie,leapdays_indexes,axis=0)
p095values_dataframe_timeserie = np.delete(p095values_dataframe_timeserie,leapdays_indexes,axis=0)
medianvalues_dataframe_timeserie = np.delete(medianvalues_dataframe_timeserie,leapdays_indexes,axis=0)
minvalues_dataframe_timeserie = np.delete(minvalues_dataframe_timeserie,leapdays_indexes,axis=0)
maxvalues_dataframe_timeserie = np.delete(maxvalues_dataframe_timeserie,leapdays_indexes,axis=0)


p005values_dataframe_timeserie_naturalyear = np.delete(p005values_dataframe_timeserie_naturalyear,natural_leapdays_indexes,axis=0)
p010values_dataframe_timeserie_naturalyear = np.delete(p010values_dataframe_timeserie_naturalyear,natural_leapdays_indexes,axis=0)
p090values_dataframe_timeserie_naturalyear = np.delete(p090values_dataframe_timeserie_naturalyear,natural_leapdays_indexes,axis=0)
p095values_dataframe_timeserie_naturalyear = np.delete(p095values_dataframe_timeserie_naturalyear,natural_leapdays_indexes,axis=0)
medianvalues_dataframe_timeserie_naturalyear = np.delete(medianvalues_dataframe_timeserie_naturalyear,natural_leapdays_indexes,axis=0)
minvalues_dataframe_timeserie_naturalyear = np.delete(minvalues_dataframe_timeserie_naturalyear,natural_leapdays_indexes,axis=0)
maxvalues_dataframe_timeserie_naturalyear = np.delete(maxvalues_dataframe_timeserie_naturalyear,natural_leapdays_indexes,axis=0)

# Calculate monthly anomalies of variables        
monthly_variablesdata = np.zeros([np.size(climatedata_df,1)-3,12,nyears],float)
monthly_anomalies = np.zeros([np.size(climatedata_df,1)-3,12,nyears],float)
months = [1,2,3,4,5,6,7,8,9,10,11,12]
months = np.tile(months,nyears)

for var in range(np.size(climatedata_df,1)-3):
    j=0
    k=-1
    for i in range(len(months)):
        if months[i]==1:
            j=0
            k=k+1
            monthly_variablesdata[var,j,k]=np.nanmean(climatedata_df.iloc[(climatedata_df.index.year==years_array[k]) & (climatedata_df.index.month==months[i]),var])
        else:
            j=j+1
            monthly_variablesdata[var,j,k]=np.nanmean(climatedata_df.iloc[(climatedata_df.index.year==years_array[k]) & (climatedata_df.index.month==months[i]),var])
       
    for j in range(12):
        for k in range(nyears):
            monthly_anomalies[var,j,k] = monthly_variablesdata[var,j,k] - np.nanmean(monthly_variablesdata[var,j,:])

monthlyvariables_df = np.reshape(monthly_variablesdata,[np.size(monthly_variablesdata,0),np.size(monthly_variablesdata,1)*np.size(monthly_variablesdata,2)],order='F')
monthlyanomalies_df = np.reshape(monthly_anomalies,[np.size(monthly_anomalies,0),np.size(monthly_anomalies,1)*np.size(monthly_anomalies,2)],order='F')
# monthlyanomalies_df = np.reshape(monthly_anomalies,[np.size(monthly_anomalies,0)*np.size(monthly_anomalies,1),1],order='F')

monthlyvariables_df = monthlyvariables_df.T
monthlyanomalies_df = monthlyanomalies_df.T


date1 = str(yearinicio)+'-'+str(1)
date2 = str(yeartoday)+'-'+str(12)
monthlyvariables_df = pd.DataFrame(monthlyvariables_df,index=pd.date_range(date1, date2,freq=pd.offsets.MonthBegin(1)).tolist(),columns=climatedata_df.columns[:-3])
monthlyanomalies_df = pd.DataFrame(monthlyanomalies_df,index=pd.date_range(date1, date2,freq=pd.offsets.MonthBegin(1)).tolist(),columns=climatedata_df.columns[:-3])


moving_anommonthly_variablesdata = np.zeros([len(monthlyanomalies_df),np.size(climatedata_df,1)-3],float)

for var in range(np.size(climatedata_df,1)-3):    
    n1 = 12       # Puntos sobre los que promediar
    if n1 % 2 == 0:
    
        K = (n1-1)//2
        
        for i in range(K,len(monthlyanomalies_df)-K):
            if np.isnan(monthlyanomalies_df.iloc[i,var]) == True:
                moving_anommonthly_variablesdata[i,var] = np.nan
            else:
                moving_anommonthly_variablesdata[i,var]=(1/(2*K+1))*monthlyanomalies_df.iloc[i-K:i+K,var].sum()
            
        moving_anommonthly_variablesdata[:K,var]=np.nan
        moving_anommonthly_variablesdata[-K:,var]=np.nan
        
    else:
        K = (n1)//2
        
        for i in range(K,len(monthlyanomalies_df)-K):
            moving_anommonthly_variablesdata[i,var]=(1/n1)*monthlyanomalies_df.iloc[i-K+1:i+K,var].sum()+((1/n1)*(0.5*monthlyanomalies_df.iloc[i-K,var]+0.5*monthlyanomalies_df.iloc[i+K,var]))
        
        moving_anommonthly_variablesdata[:K,var]=np.nan
        moving_anommonthly_variablesdata[-K:,var]=np.nan  


#monthlyanomalies_df_index_mod = monthlyanomalies_df.index + dt.timedelta(weeks=3.5*K) 

        
        
# To visualize data of a certain period of days

# daytoday = int(input('Insert the julian day of the last data (if not known insert NA): '))
dia_visualizador = int(input('Input the day of the initial date of the period to analize (to skip insert NA): '))
mes_visualizador = int(input('Input the month of the initial date of the period to analize: '))
if monthtoday!='NA':
    periododias = int(input('Input the period of days to analize: '))
    climatedata_reshaped_naturalyear_period = np.zeros([np.size(climatedata_df,1)-3,nyears],float)
    p005values_dataframe_period = np.zeros([np.size(climatedata_df,1)-3,nyears],float)
    p010values_dataframe_period = np.zeros([np.size(climatedata_df,1)-3,nyears],float)
    p090values_dataframe_period = np.zeros([np.size(climatedata_df,1)-3,nyears],float)
    p095values_dataframe_period = np.zeros([np.size(climatedata_df,1)-3,nyears],float)
    meanvalues_dataframe_period = np.zeros([np.size(climatedata_df,1)-3,nyears],float)
    medianvalues_dataframe_period = np.zeros([np.size(climatedata_df,1)-3,nyears],float)
    minvalues_dataframe_period = np.zeros([np.size(climatedata_df,1)-3,nyears],float)
    maxvalues_dataframe_period = np.zeros([np.size(climatedata_df,1)-3,nyears],float)
    
    
    for var in range(0,np.size(climatedata_df,1)-3):
        for i in range(0,nyears):
            for j in range(0,366):
                if climatedata_reshaped_naturalyear[var,j,-2]==dia_visualizador and climatedata_reshaped_naturalyear[var,j,-1]==mes_visualizador:
                    climatedata_reshaped_naturalyear_period[var,i] = np.nanmean(climatedata_reshaped_naturalyear[var,j:(j+periododias+1),i])
                    maxvalues_dataframe_period[var,i] = np.nanmax(climatedata_reshaped_naturalyear[var,j:(j+periododias+1),i])
                    minvalues_dataframe_period[var,i] = np.nanmin(climatedata_reshaped_naturalyear[var,j:(j+periododias+1),i])
    
        p005values_dataframe_period[var,:] = np.tile(np.nanpercentile(climatedata_reshaped_naturalyear_period[var,:],5),nyears)            
        p010values_dataframe_period[var,:] = np.tile(np.nanpercentile(climatedata_reshaped_naturalyear_period[var,:],10),nyears)            
        p090values_dataframe_period[var,:] = np.tile(np.nanpercentile(climatedata_reshaped_naturalyear_period[var,:],90),nyears)            
        p095values_dataframe_period[var,:] = np.tile(np.nanpercentile(climatedata_reshaped_naturalyear_period[var,:],95),nyears) 
        meanvalues_dataframe_period[var,:] = np.tile(np.nanmean(climatedata_reshaped_naturalyear_period[var,:]),nyears)            
        medianvalues_dataframe_period[var,:] = np.tile(np.nanmedian(climatedata_reshaped_naturalyear_period[var,:]),nyears)            
           
            
            
months = np.concatenate([['Jan'],['Feb'],['Mar'],['Apr'],['May'],['Jun'],['Jul'],['Aug'],['Sep'],['Oct'],['Nov'],['Dec']])


#### OPTIONAL: WINDOW TRENDS. ¡TAKES A LONG TIME!

# Calculate window trends

# For guarantee consistency, first and last year of data are excluded for the trends, 
# because rarely dataseries start at January 1st and if last year of data is not finished trends wouldn't be representative

p_mktest_variables = np.zeros([np.size(climatedata_df,1)-3,nyears-3,nyears-3],float)
p_mktest_variables[:] = np.nan

variables_trends = np.zeros([np.size(climatedata_df,1)-3,nyears-3,nyears-3],float)
variables_trends[:] = np.nan


annual_variabledata = np.zeros([np.size(climatedata_df,1)-3,nyears-2],float)

variabledata_yeartoyearchanges = np.zeros([np.size(climatedata_df,1)-3,nyears-3,nyears-3],float)
variabledata_yeartoyearchanges[:] = np.nan

for var in range(np.size(climatedata_df,1)-3):
    for i in range(1,nyears-1):
        annual_variabledata[var,i-1] = climatedata_df_complete.iloc[climatedata_df_complete.index.year==years_array[i],var].mean()


window_years = np.arange(yearinicio+1,yeartoday,1)

for var in range(np.size(climatedata_df,1)-3):
    for i in range(0,nyears-3):
        for j in range(0,nyears-3):
            if i>j:
                continue
            else:
                variables_trends[var,i,j] = np.polyfit(window_years[j-i:j+2],annual_variabledata[var,j-i:j+2],1)[0]
                p_mktest_variables[var,i,j] = np.sign(variables_trends[var,i,j])*mk.original_test(climatedata_df_complete.iloc[(climatedata_df_complete.index.year>=yearinicio+1+j-i) & (climatedata_df_complete.index.year<=yearinicio+1+j+1),var])[2]
                variabledata_yeartoyearchanges[var,i,j] = variables_trends[var,i,j]*(len(window_years[j-i:j+2])-1)

    

## Some yearly data linear trends. By default, exclude first and last years as they usually are incomplete.
# To change this, change "nyears-2" for "nyears-1" if there is only one year with missing data, or "nyears" if every year is complete
linreg_variable = np.zeros([nyears-2,np.size(climatedata_df,1)-3],float)
variable_trend = np.zeros([nyears-2,np.size(climatedata_df,1)-3],float)
variable_confidenceinterval_high = np.zeros([nyears-2,np.size(climatedata_df,1)-3],float)
variable_confidenceinterval_low = np.zeros([nyears-2,np.size(climatedata_df,1)-3],float)

yearly_data = np.zeros([nyears-2,np.size(climatedata_df,1)-3],float)
for var in range(np.size(climatedata_df,1)-3):
    for j in range(nyears-2):
        yearly_data[j,var] = climatedata_df.iloc[climatedata_df.index.year==years_array[j+1],var].mean()


for var in range(np.size(climatedata_df,1)-3):
    linreg_variable[:,var],variable_trend[:,var],variable_confidenceinterval_high[:,var],variable_confidenceinterval_low[:,var], alpha = linear_regression(years_array[1:-1],yearly_data[:,var])

# Correlation between different yearly variables
yearly_correlations = np.zeros([np.size(climatedata_df,1)-3,np.size(climatedata_df,1)-3],float)

for var in range(np.size(climatedata_df,1)-3):
    for var1 in range(np.size(climatedata_df,1)-3):
        yearly_correlations[var,var1] = np.corrcoef(yearly_data[:,var],yearly_data[:,var1])[1,0]




# =============================================================================
#  Some example plots
# =============================================================================

variables_to_plot = [0,1,2] # SELECT VARIABLES TO PLOT 
colours = ['b','r','k']
for i in variables_to_plot:
    fig, ax = plt.subplots(figsize=(15,7))
    ax.plot(climatedata_df.index, climatedata_df.iloc[:,i],color=colours[i],marker='o',fillstyle='full',markersize='6')
    ax.plot(climatedata_df.index, np.tile(np.nanmin(minvalues_dataframe_timeserie[:,i]),len(climatedata_df.index)),'b--',label='Record min.: %.1fºC' %np.nanmin(minvalues_dataframe_timeserie[:,i]))
    ax.plot(climatedata_df.index, np.tile(np.nanmax(maxvalues_dataframe_timeserie[:,i]),len(climatedata_df.index)),'r--',label='Record max.: %.1fºC' %np.nanmax(maxvalues_dataframe_timeserie[:,i]))
    ax.plot(climatedata_df.index, moving_variables.iloc[:,i],color='cyan',label='%i-MA' %n1)
    ax.plot(climatedata_df.index, medianvalues_dataframe_timeserie[:len(climatedata_df.index),i],color='g',label='Median')
    ax.fill_between(climatedata_df.index,p090values_dataframe_timeserie[:len(climatedata_df.index),i],p010values_dataframe_timeserie[:len(climatedata_df.index),i],color='grey',alpha=0.5,label="10%-90%")
    ax.fill_between(climatedata_df.index,p095values_dataframe_timeserie[:len(climatedata_df.index),i],p005values_dataframe_timeserie[:len(climatedata_df.index),i],color='grey',alpha=0.25,label="5%-95%")
    ax.grid(color='black',alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('%s' %climatedata_df.columns[i])
    ax.set_title('%s in %s (%s - %s)' %(climatedata_df.columns[i],location,climatedata_df.index[0],climatedata_df.index[-1]))
    ax.legend(bbox_to_anchor=(0.98, -0.10),ncol=2,fontsize=12).set_visible(True)
    fig.autofmt_xdate()
    if daytoday !='NA':
        fig, ax = plt.subplots(figsize=(15,7))
        ax.plot(years_array[:-1], climatedata_reshaped_naturalyear_period[i,:-1],'ko',fillstyle='full',markersize='8')
        ax.plot(years_array[-1], climatedata_reshaped_naturalyear_period[i,-1],color='orange',marker='o',fillstyle='full',markersize='8',label='%s: %.1fºC' %(years_array[-1],climatedata_reshaped_naturalyear_period[i,-1]))
        ax.plot(years_array, medianvalues_dataframe_period[i,:],color='g',label='Median: %.1f' %medianvalues_dataframe_period[i,-1])
        ax.fill_between(years_array,p090values_dataframe_period[i,:],p010values_dataframe_period[i,:],color='grey',alpha=0.5,label="10%-90%")
        ax.fill_between(years_array,p095values_dataframe_period[i,:],p005values_dataframe_period[i,:],color='grey',alpha=0.25,label="5%-95%")
        ax.plot(years_array, np.tile(np.nanmin(climatedata_reshaped_naturalyear_period[i,:]),len(years_array)),'b--',label='Record min.: %.1fºC' %np.nanmin(climatedata_reshaped_naturalyear_period[i,:]))
        ax.plot(years_array, np.tile(np.nanmax(climatedata_reshaped_naturalyear_period[i,:]),len(years_array)),'r--',label='Record max.: %.1fºC' %np.nanmax(climatedata_reshaped_naturalyear_period[i,:]))
        ax.grid(color='black',alpha=0.5)
        ax.set_xlabel('Año',fontsize=12)
        ax.set_ylabel('Temperatura (ºC)',fontsize=12)
        ax.set_title('Temperaturas máximas en Campillos. Periodo %.0f-%s + %.0f días posteriores' %(int(dia_visualizador),months[mes_visualizador-1],periododias-1),fontsize=16)
        ax.set_title('%s in %s for the period %i-%s + %i days' %(climatedata_df.columns[i],location,int(dia_visualizador),months[mes_visualizador-1],periododias-1))
        plt.legend(bbox_to_anchor=(1.08, -0.10),ncol=2,fontsize=12).set_visible(True)
        fig.autofmt_xdate()

    # Plotting one temperature variable and other meteorological variable at the same time. Adjust indexes depending on your data
    if i == 2:  
        fig, ax = plt.subplots(figsize=(15,7))
        ax.plot(climatedata_df.index, climatedata_df.iloc[:,i],color=colours[i],marker='o',fillstyle='full',markersize='6')
        ax.bar(climatedata_df.index, climatedata_df.iloc[:,i+2],color='grey',alpha=0.8)
        ax.grid(color='black',alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('%s' %climatedata_df.columns[i])
        ax.set_title('%s in %s (%s - %s)' %(climatedata_df.columns[i],location,climatedata_df.index[0],climatedata_df.index[-1]))
        ax.legend(bbox_to_anchor=(0.98, -0.10),ncol=2,fontsize=12).set_visible(True)
        fig.autofmt_xdate()
        
    # Probability distribution
    fig=plt.figure(figsize=(15,10))
    ax=fig.add_subplot(111)
    climatedata_df.iloc[:,i].plot(kind='kde',subplots=True,ax=ax,color=colours[i])
    ax.set_title('%s probability distribution in %s' %(climatedata_df.columns[i],location))
    ax.set_ylabel('Probability (1=100%)')
    ax.set_xlabel('%s' %climatedata_df.columns[i])
    ax.set_xlim([-17,47])
    ax.grid(color='black',alpha=0.5)
    ax.axvline(x=climatedata_df.iloc[-1,i],color='orange',linestyle='--')
    ax.legend(['%s-%s' %(climatedata_df.index.year[0],climatedata_df.index.year[-1]),'Last data'])
    
    # Seasonal probability distributions
    fig=plt.figure(figsize=(15,10))
    ax=fig.add_subplot(221)
    climatedata_df.iloc[(climatedata_df.index.month<3) | (climatedata_df.index.month==12),i].plot(kind='kde',subplots=True,ax=ax,color='blue')
    ax.set_title('%s winter probability distribution in %s' %(climatedata_df.columns[i],location))
    ax.set_ylabel('Probability (1=100%)')
    ax.set_xlabel('%s' %climatedata_df.columns[i])
    ax.set_xlim([-17,47])
    ax.grid(color='black',alpha=0.5)
    ax.axvline(x=climatedata_df.iloc[-1,i],color='orange',linestyle='--')
    ax.legend(['%s-%s'%(climatedata_df.index.year[0],climatedata_df.index.year[-1]),'Last data'])
    ax=fig.add_subplot(222)
    climatedata_df.iloc[(climatedata_df.index.month>=3) & (climatedata_df.index.month<6),i].plot(kind='kde',subplots=True,ax=ax,color='tab:green')
    ax.set_title('%s spring probability distribution in %s' %(climatedata_df.columns[i],location))
    ax.set_ylabel('Probability (1=100%)')
    ax.set_xlabel('%s' %climatedata_df.columns[i])
    ax.set_xlim([-17,47])
    ax.grid(color='black',alpha=0.5)
    ax.axvline(x=climatedata_df.iloc[-1,i],color='orange',linestyle='--')
    ax.legend(['%s-%s'%(climatedata_df.index.year[0],climatedata_df.index.year[-1]),'Last data'])
    ax=fig.add_subplot(223)
    climatedata_df.iloc[(climatedata_df.index.month>=6) & (climatedata_df.index.month<9),i].plot(kind='kde',subplots=True,ax=ax,color='tab:red')
    ax.set_title('%s summer probability distribution in %s' %(climatedata_df.columns[i],location))
    ax.set_ylabel('Probability (1=100%)')
    ax.set_xlabel('%s' %climatedata_df.columns[i])
    ax.set_xlim([-17,47])
    ax.grid(color='black',alpha=0.5)
    ax.axvline(x=climatedata_df.iloc[-1,i],color='orange',linestyle='--')
    ax.legend(['%s-%s'%(climatedata_df.index.year[0],climatedata_df.index.year[-1]),'Last data'])
    ax=fig.add_subplot(224)
    climatedata_df.iloc[(climatedata_df.index.month>=9) & (climatedata_df.index.month<12),i].plot(kind='kde',subplots=True,ax=ax,color='tab:purple')
    ax.set_title('%s autumn probability distribution in %s' %(climatedata_df.columns[i],location))
    ax.set_ylabel('Probability (1=100%)')
    ax.set_xlabel('%s' %climatedata_df.columns[i])
    ax.set_xlim([-17,47])
    ax.grid(color='black',alpha=0.5)
    ax.axvline(x=climatedata_df.iloc[-1,i],color='orange',linestyle='--')
    ax.legend(['%s-%s'%(climatedata_df.index.year[0],climatedata_df.index.year[-1]),'Last data'])
                            
    
    
    # Plot window trends
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    
    cmap = mpl.colors.ListedColormap(['cyan','blue',
                                      'red','orange'])
    cmap.set_over('yellow',alpha=0.5)
    cmap.set_under('darkgrey',alpha=0.5)
    
    bounds = [-0.05, -0.01, 0.0, 0.01, 0.05]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(15,7))
    ax = plt.gca()
    im=ax.matshow(p_mktest_variables[i,:,:],cmap=cmap,norm=norm,clim=[-0.05,0.05],aspect='auto')
    #ax.contour(p_mktest_minimas, levels=[-0.05, -0.01, 0.01, 0.05], colors='black',linewidths=1)
    ax.set_title('Mann-Kendall test significancy level of %s annual trends in %s' %(climatedata_df_complete.columns[i],location),fontsize=16)
    ax.set_xlabel('Last year of the window',fontsize=12)
    ax.set_ylabel('Window length (years)',fontsize=12)
    ax.set_yticks(np.arange(0,nyears-1,1))
    ax.set_yticklabels(np.arange(1,nyears,1))
    ax.set_ylim([nyears-0.5,-0.5])
    ax.set_xticklabels(['0','2012','2013','2014','2015','2016','2017','2018','2019','2020'])
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=False)
    cbar=fig.colorbar(im)
    cbar.set_ticks([-0.05,-0.01,0.01,0.05])
    cbar.set_label('p-value', rotation=270)

   
    cmap = mpl.colors.ListedColormap(['#642B87','#473E99','#2D4A9F','#2D4A9F','#006E6F','#00A651','#2FB562','#C0E2C7',
                                      '#FBF9CA','#F6EF27','#F2EB0B','#FFDB00','#FDB414','#F58024','#EF482D','#EE292F'])
    cmap.set_over('#FF196E',alpha=0.5)
    cmap.set_under('#641285',alpha=0.5)
    
    bounds = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(15,7))
    ax = plt.gca()
    im=ax.matshow(variables_trends[i,:,:],cmap=cmap,norm=norm,aspect='auto')
    #ax.contour(tendencia_maximas, levels=bounds, colors='black',linewidths=1)
    ax.set_title('%s annual trends in %s' %(climatedata_df_complete.columns[i],location),fontsize=16)
    ax.set_xlabel('Last year of the window',fontsize=12)
    ax.set_ylabel('Window length (years)',fontsize=12)
    # ax.xaxis.set_major_locator(MultipleLocator(5))
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)
    ax.set_yticks(np.arange(0,nyears-1,1))
    ax.set_yticklabels(np.arange(1,nyears,1))
    ax.set_ylim([nyears-0.5,-0.5])
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=False)
    ax.set_xticks(np.arange(0,nyears-2,1))
    ax.set_xticklabels(np.arange(yearinicio+2,yearinicio+nyears-1,1))
    # ax.xaxis.set_major_locator(MultipleLocator(5))
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    cbar=fig.colorbar(im,extend='both')
    cbar.set_label('variation/year')

    # Timeseries of variable anomalies
    fig, ax = plt.subplots(figsize=(15,7))
    for j in range(0,len(monthlyvariables_df)):
        if monthlyanomalies_df.iloc[j,i]>0:
            ax.bar(monthlyvariables_df.index[j], monthlyanomalies_df.iloc[j,i],color='tab:red',width=21)
        else:
            ax.bar(monthlyvariables_df.index[j], monthlyanomalies_df.iloc[j,i],color='tab:blue',width=21)
    ax.plot(monthlyanomalies_df.index,moving_anommonthly_variablesdata[:,i],color='black',label='%i-month moving average' %n1)
    ax.grid(color='black',alpha=0.5)
    ax.set_ylabel('Temperature anomaly (ºC)')
    ax.set_title('Monthly %s anomaly in %s' %(climatedata_df.columns[i],location), fontsize=16)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    # Get only the year to show in the x-axis:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.text(0.15, 0.06, 'Alejandro Rodríguez Sánchez. @alexrs197 @MeteoCampillos', fontsize=10, transform=plt.gcf().transFigure)
    plt.text(0.74, 0.955, 'Climate normal period: %s-%s' %(climatedata_df.index.year[0],climatedata_df.index.year[-1]), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.74, 0.93, 'Database: SIAR', fontsize=12, transform=plt.gcf().transFigure)
    fig.autofmt_xdate()
    ax.legend(fontsize=9,loc='best').set_visible(True)
    
    # Monthly values by year
    from matplotlib.dates import date2num    
    years_dt =  pd.date_range(start=str(climatedata_df.index.year[0]-1)+'-01-01', end=str(climatedata_df.index.year[-1])+'-01-01', freq='Y')
    years_dt = date2num(years_dt)
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    barwidth = 25
    ax.bar(years_dt-barwidth*11/2,monthly_variablesdata[i,0,:], width=barwidth,color='dodgerblue',label='January')
    ax.bar(years_dt-barwidth*9/2,monthly_variablesdata[i,1,:], width=barwidth,color='cornflowerblue',label='February')
    ax.bar(years_dt-barwidth*7/2,monthly_variablesdata[i,2,:],width=barwidth,color='lime',label='March')
    ax.bar(years_dt-barwidth*5/2,monthly_variablesdata[i,3,:], width=barwidth,color='green',label='April')
    ax.bar(years_dt-barwidth*3/2,monthly_variablesdata[i,4,:], width=barwidth,color='mediumseagreen',label='May')
    ax.bar(years_dt-barwidth*1/2,monthly_variablesdata[i,5,:], width=barwidth,color='olive',label='June')
    ax.bar(years_dt+barwidth*1/2,monthly_variablesdata[i,6,:], width=barwidth,color='orange',label='July')
    ax.bar(years_dt+barwidth*3/2,monthly_variablesdata[i,7,:],width=barwidth,color='red',label='August')
    ax.bar(years_dt+barwidth*5/2,monthly_variablesdata[i,8,:], width=barwidth,color='blueviolet',label='September')
    ax.bar(years_dt+barwidth*7/2,monthly_variablesdata[i,9,:], width=barwidth,color='mediumpurple',label='October')
    ax.bar(years_dt+barwidth*9/2,monthly_variablesdata[i,10,:],width=barwidth,color='orchid',label='November')
    ax.bar(years_dt+barwidth*11/2,monthly_variablesdata[i,11,:], width=barwidth,color='blue',label='December')
    #for i in range(len(years_dt)):
    #    ax.text(years_dt[i],mediana_viento_septiembre[i+1]+0.1,(mediana_viento_septiembre[i+1]),fontsize=10,fontweight='bold')
    ax.grid(color='black',alpha=0.5)
    ax.set_xlabel('Date (year)')
    ax.set_ylabel('Monthly mean' )
    ax.set_title('%s monthly mean in %s' %(climatedata_df.columns[i],location))
    ax.legend(ncol=2,loc='best').set_visible(True)
    ax.xaxis_date()
    fig.autofmt_xdate()

    # Highlight one month
    # Monthly values by year
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    ax.bar(years_dt-barwidth*11/2,monthly_variablesdata[i,0,:], width=barwidth,color='darkgrey',label='January')
    ax.bar(years_dt-barwidth*9/2,monthly_variablesdata[i,1,:], width=barwidth,color='darkgrey',label='February')
    ax.bar(years_dt-barwidth*7/2,monthly_variablesdata[i,2,:],width=barwidth,color='darkgrey',label='March')
    ax.bar(years_dt-barwidth*5/2,monthly_variablesdata[i,3,:], width=barwidth,color='darkgrey',label='April')
    ax.bar(years_dt-barwidth*3/2,monthly_variablesdata[i,4,:], width=barwidth,color='darkgrey',label='May')
    ax.bar(years_dt-barwidth*1/2,monthly_variablesdata[i,5,:], width=barwidth,color='olive',label='June')
    ax.bar(years_dt+barwidth*1/2,monthly_variablesdata[i,6,:], width=barwidth,color='darkgrey',label='July')
    ax.bar(years_dt+barwidth*3/2,monthly_variablesdata[i,7,:],width=barwidth,color='darkgrey',label='August')
    ax.bar(years_dt+barwidth*5/2,monthly_variablesdata[i,8,:], width=barwidth,color='darkgrey',label='September')
    ax.bar(years_dt+barwidth*7/2,monthly_variablesdata[i,9,:], width=barwidth,color='darkgrey',label='October')
    ax.bar(years_dt+barwidth*9/2,monthly_variablesdata[i,10,:],width=barwidth,color='darkgrey',label='November')
    ax.bar(years_dt+barwidth*11/2,monthly_variablesdata[i,11,:], width=barwidth,color='darkgrey',label='December')
    ax.grid(color='black',alpha=0.5)
    ax.set_xlabel('Date (year)')
    ax.set_ylabel('Monthly mean' )
    ax.set_title('%s monthly mean in %s' %(climatedata_df.columns[i],location))
    ax.legend(ncol=2,loc='best').set_visible(True)
    ax.xaxis_date()
    fig.autofmt_xdate()

    # Plot trends
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(211)
    ax.plot(years_array[1:-1],yearly_data[:,i],color=colours[i],label='%s' %climatedata_df.columns[i])
    ax.plot(years_array[1:-1],linreg_variable[:,i],color='green',label='Linear trend')
    ax.fill_between(years_array[1:-1],variable_confidenceinterval_high[:,i],variable_confidenceinterval_low[:,i],color='grey',alpha=0.4, label='%i%% confidence interval' %(100*(1-alpha)))
    ax.grid(color='black',alpha=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('%s' %climatedata_df.columns[i])
    ax.set_title('Yearly mean of %s with linear trend' %climatedata_df.columns[i])
    ax.legend().set_visible(True)
    fig.autofmt_xdate()

