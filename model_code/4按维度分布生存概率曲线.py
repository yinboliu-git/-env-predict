from pipeline import one_to_one_factor,one_factor,latitude_line_angle,latitude_line_block,latitude_line_angle_block
import numpy as np
import pandas as pd

if __name__ == '__main__':
    file_name = ['./results/corr_ml/gbrt-sites_scores2020s.csv',
                 './results/corr_ml/gbrt-sites_scores2030s.csv',
                 './results/corr_ml/gbrt-sites_scores2050s.csv',
                 './results/corr_ml/gbrt-sites_scores2070s.csv',]
    file_name2 = ['./results/corr_dnn_100e/fnn-sites_scores2020s.csv',
                 './results/corr_dnn_100e/fnn-sites_scores2030s.csv',
                 './results/corr_dnn_100e/fnn-sites_scores2050s.csv',
                 './results/corr_dnn_100e/fnn-sites_scores2070s.csv',]

    label = ['NOW',
             '2030s',
             '2050s',
             '2070s',]
    na = 'ml'
    # latitude_line_angle(file_name, label=label, _angle=1)
    # latitude_line_block(file_name, label=label, _block=200)
    for _angle in [0.4, 1]:
        # latitude_line_angle_block(file_name, label=label,_angle=_angle, _angle_block=10000, save_n='ml')
        latitude_line_angle_block(file_name2, label=label, _angle=_angle, _angle_block=10000, save_n='dl')

