# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:57:16 2019

@author: eleon
"""

import numpy as np
import pandas as pd
import re
import scipy
from matplotlib import pyplot as plt

def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

#df[df.applymap(isnumber)]

def loadfactor_monthly():
    df=pd.read_excel(r'LoadFactor.xls')
    df[df.applymap(isnumber)]

    data=df.values
    #print(data[:,2])
    months=np.array(data[:,1])
    load=np.array(data[:,2]);
    year=np.array(data[:,0]);
    #print(year)
    #print(isnumber(months[1]))
    i=0;
    while i < months.shape[0]:
        if isnumber(months[i])==False or np.isnan(months[i]):
            months=np.delete(months,(i),axis=0);
            load=np.delete(load,(i),axis=0);
            year=np.delete(year,(i),axis=0);
        i+=1

    bestfit_params=np.empty([1,2]);
    std_vals=np.zeros([12]);

    for i_month in range(std_vals.shape[0]):
        
        data_pt=np.empty([0]);
        year_val=np.empty([0]);
        for i in range(months.shape[0]):
            if months[i]==i_month+1:
                data_pt=np.append(data_pt,load[i]);
                year_val=np.append(year_val,year[i])
        #plt.plot(year_val,data_pt)
        #plt.show()
        params=np.polyfit(year_val,data_pt,1);
#        plt.plot(year_val,data_pt,label='Real World Data')
#        plt.plot(year_val,params[0]*year_val+params[1],label='Best Fit');
#        plt.xlabel('Year');
#        plt.ylabel('Load Factor of the month');
#        plt.title('Load Factor for the month of January');
#        plt.show()
        if i_month==0:
            bestfit_params[:]=params[:]
            std_vals[i_month]=np.std(data_pt);
            continue;
        bestfit_params=np.vstack([bestfit_params,params])
        std_vals[i_month]=np.std(data_pt);
#print(bestfit_params.shape,std_vals.shape)

    return [bestfit_params,std_vals]
    '''
    s=np.random.normal(2019*bestfit_params[6,0]+bestfit_params[6,1],std_vals[0],10000)
    plt.hist(s,1000,normed=True);
    plt.xlabel('Load Factor');
    plt.ylabel('Probability');
    plt.title('Load Factor Distribution for July 2020');
    plt.show()
    '''
def get_sample(time,mean,std):

    return np.random.normal(time*mean[0,0]+mean[0,1],std[0]);

def fuel_data():
    df=pd.read_excel(r'fuel_data.xlsx')
    df[df.applymap(isnumber)]

    data=df.values
    months=np.array(data[:,1])
    mean=np.array([data[:12,3],data[12:,3]]);
    stdv=np.array([data[:12,7],data[12:,7]]);
    return mean, stdv

mean,stdv=fuel_data();

def fuel(month,year):
   
    global mean
    global stdv
    if month>=12:
        month=int(float(month)/12.)
        year+=1
    
    #price=np.random.normal(mean[year,month],stdv[year,month])
    price=mean[year,month]
    while price<0:
        #price=np.random.normal(mean[year,month],stdv[year,month])
        price=mean[year,month]
    return price

