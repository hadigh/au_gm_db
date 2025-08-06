import sys
import pandas as pd
from obspy import read

"""
A Python script that augments the AU waveform list (au_wf_lst) file with network, channel, location, and sampling rate 
information that is retrieved from the mseed files. It would merge the traces for each channels, if required! 
The output is stored as a csv file labelled as extended_au_wf_lst!

The script lists the problematic streams that contain trace segments (of a channel) that cannot be merged! and exports
them to a log_file "Not_mergable_streams.txt"

Parameters

Inputs:

wf_lst_file: The path to the CSV file containing the available waveform names, as well as information related to 
the corresponding earthquake and recording station parameters

wf_dir: .The directory path where the waveform files in MiniSEED format are stored.

Outputs:

mode_wf_lst_file: The file path for the augmented CSV file, which includes information 
on network, channel, location, and sampling rate, in addition to that from the original waveform list file. 
"""
# Inputs
wf_dir = "../inputs/wf_data"
wf_lst_file = "../inputs/au_wf_lst.csv"
# wf_lst_file = "../inputs/2021_WP_wf_lst.csv"  #######this is just to read certain event!


# Outputs
mod_wf_lst_file = "../outputs/extended_au_wf_lst.csv"
# mod_wf_lst_file = (
#     "../outputs/extended_2021_WP_wf_lst.csv"  # this is just for one event!
# )
log_file_path = "../outputs/not_mergable_streams.log"

# modules
def get_mseed_file_name(mseed_orig_path, wf_path=wf_dir):
    return "/".join((wf_path, mseed_orig_path.split("/")[-1]))


# add the file name of waveforms to the wf_lst csv file!
wf_lst = pd.read_csv(wf_lst_file)
wf_lst["mseed_path"] = wf_lst["MSEEDFILE"].apply(get_mseed_file_name)
wf_files = wf_lst["mseed_path"]

mod_wf_lst = pd.DataFrame()
log_entries = []
for index, row in wf_lst.iterrows():
    f = row["mseed_path"]
    st = read(f)
    
    try:
        # # Initialize a flag to check if all traces have the same start and end times
        # identify if the stream is problematic or not!
        # stream is problematic if it has two traces with same loc.cha.sample_rate_starttime_endtime but different waveforms (different PGAs)
        # if stream is problematic do not merge, if not merge!
        dum = {
            "cha": [],
            "loc": [],
            "sr": [],
            "starttime": [],
            "endtime": [],
            "pga": [],
        }

        for trace in st:
            dum["cha"].append(trace.stats.channel)
            dum["loc"].append(trace.stats.location)
            dum["sr"].append(trace.stats.sampling_rate)
            dum["starttime"].append(str(trace.stats.starttime))
            dum["endtime"].append(str(trace.stats.endtime))
            dum["pga"].append(max(abs(trace.data)))

        df = pd.DataFrame.from_dict(dum)
        grouped = df.groupby(["cha", "loc", "sr", "starttime", "endtime"])[
            "pga"
        ].nunique()
        inconsistent_groups = grouped[grouped > 1]

        
        
        if len(inconsistent_groups) > 0:
            
            log_entries.append(f"\n--- Problematic Stream Detected ---")
            log_entries.append("Inconsistent Groups:")
            log_entries.append(inconsistent_groups.to_string())
            log_entries.append("Full Stream:")
            log_entries.append(str(st))
            log_entries.append("--- End of Stream ---\n")
            print(f"Problematic traces logged - repeated channel!")
    
            st.merge(method=-1)

        else:
            # apply default merge!
            st.merge(method=0, fill_value="interpolate")

    except:
        # print("could not merge data for %s" % st)
        
        log_entries.append(f"\n--- Exception during merge ---")
        log_entries.append("Stream:")
        log_entries.append(str(st))
        log_entries.append("--- End of Stream ---\n")

        print(f"Problematic traces logged - other merge issues!")
        


    # add the instrument type, location code and sampling_rate from mseed file to the wf_lst
    # if colocated would add extra rows for each instrument code!
    for tr in st:
        if not tr.stats.channel[0] in ["L", "V", "W"]:
            row["STA_ORIG"] = row["STA"]
            row["NET"] = tr.stats.network
            row["STA"] = tr.stats.station
            row["LOC"] = tr.stats.location
            row["CHA"] = tr.stats.channel
            row["SAMPLING_RATE"] = tr.stats.sampling_rate
            mod_wf_lst = pd.concat([mod_wf_lst, row], axis=1, ignore_index=False)

mod_wf_lst = mod_wf_lst.transpose()
mod_wf_lst.reset_index()

mod_wf_lst.to_csv(mod_wf_lst_file)

# Write to log file once
with open(log_file_path, "a") as f:
    f.write("\n".join(log_entries))

