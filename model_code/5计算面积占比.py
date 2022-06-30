from pipeline import get_area_
import numpy as np
import pandas as pd

if __name__ == '__main__':
    data = 'corr_dnn_100e/fnn-'
    name = ['./results/'+data+'sites_4class2020s.csv',
            './results/'+data+'sites_4class2030s.csv',
            './results/'+data+'sites_4class2050s.csv',
            './results/'+data+'sites_4class2070s.csv',]
    get_area_(name,960.3)

