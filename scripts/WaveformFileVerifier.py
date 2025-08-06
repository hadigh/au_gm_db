import pandas as pd
from glob import glob
from shutil import copy

"""
Python script that verifies the existence of waveform files listed in `au_wf_lst.csv` within `wf_dir`. 

"""

wf_lst_file = "../inputs/au_wf_lst.csv"
# wf_lst_file = "../inputs/tmp_au_wf_lst.csv"  #######this should change when Trev finds the missing mseed files!
# wf_lst_file = "../inputs/2021_WP_wf_lst.csv"  #######this is just to read certain event!


wf_dir = "../inputs/wf_data"

wf_lst = pd.read_csv(wf_lst_file)

print("These mseed files are missing:")
for f_name in wf_lst["MSEEDFILE"]:

    mseed_dir = f_name.split("/")[0]
    mseed_file = f_name.split("/")[-1]
    f = "/".join(("../inputs/wf_data", mseed_file))
    if not glob(f):
        
        print(f_name)
