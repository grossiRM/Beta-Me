import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyemu
def main():

    try:
       os.remove(r'heads.csv')
    except Exception as e:
       print(r'error removing tmp file:heads.csv')
    try:
       os.remove(r'sfr.csv')
    except Exception as e:
       print(r'error removing tmp file:sfr.csv')
    try:
       os.remove(r'heads.tdiff.csv')
    except Exception as e:
       print(r'error removing tmp file:heads.tdiff.csv')
    try:
       os.remove(r'sfr.tdiff.csv')
    except Exception as e:
       print(r'error removing tmp file:sfr.tdiff.csv')
    pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv',chunk_len=50)

if __name__ == '__main__':
    mp.freeze_support()
    main()

