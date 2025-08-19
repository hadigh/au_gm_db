from datetime import datetime
import pandas as pd
from obspy import read_inventory
from obspy.io.sac import sacpz
from obspy.core import inventory, Trace
import numpy as np
from obspy.geodetics.base import gps2dist_azimuth


"""
Python script that merges the AU waveform list (extended_au_wf_lst) file with Station list (stationlist) file as well as

station Vs30 file! (the average Vs30 is also computed for each station based on assumed weights for each approach! see the function below)

Parameters

Inputs:

wf_lst_file: The path to the CSV file containing the list of available waveform with extended info on network, 
channel, location, and sampling rate (i.e. extended_au_wf_lst.csv output of WaveformListAugmenter.py) 

sta_lst_file: path to the stationlist.dat (TA file on station-metadata)

inv_AUSPASS_II_IU.xml: Station xml inventory that is compiled from dataless files available for listed stations!
# it is the output of merge_ANU_II_IU_S1.py (from hard disk!)

sta_vs_file: list of Vs30 values (from different approaches) for recording stations!

Outputs:

mode_wf_lst_file: The file path for the merged waveform list with station list, it also specifies the source for 
the recording station metadata and if such information is missing! 
(merged_au_wf_lst.csv: contain information on both waveforms and recording station)

LOGFILE: report_duplicated_channels: 
list waveforms with possible channel code issues by listing those waveforms with same:
"STA", "CHA", "SAMPLING_RATE", "ORIGIN", "LOC" (dont understand it yet should go back to the code later!)

LOGFILE: report_network_log:

LOGFILE: report_au_sta_with_multi_loc_code: list AU stations, same station, same channel, different location code
this may (or may not) cause ambiguity in getting response from TA stationlist file which does not include location code!
perhaps the report_duplicated_channel would be more accurate to identify issues!


LOGFILE: paz_map_file: summary of the applied channel code changes (to help with TA quick review)

LOGFILE: latlon_mismatch_file: list records with same network, same station, but different lat and long! (ideally this should not be the case)


LOGFILE: stacode_mismatch_file: list records that there is a mismatch between the original station code that is listed in
au_wf_lst.csv and the one retrieved from mseed file itself!
"""
# Inputs
wf_lst_file = "../outputs/extended_au_wf_lst.csv"
sta_lst_file = "../inputs/stationlist.dat"
inv_AUSPASS_II_IU_file = "../outputs/inv_AUSPASS_II_IU.xml"
inv_AU_file = "../inputs/AU.dataless"
sta_vs_file = "../inputs/au_station_vs30.csv"

# Outputs
mod_wf_lst_file = "../outputs/merged_au_wf_lst.csv"
report_au_sta_vs = "../outputs/sta_vs_log.log"
report_duplicated_channels = "../outputs/channel_duplicates.log"
report_network_log = "../outputs/network_patching_log.log"
report_au_sta_with_multi_loc_code = "../outputs/au_sta_multi_loc.log"
report_cha_rev = "../outputs/channel_code_revision.log"
paz_map_file = "../outputs/paz_map_cha.log"
latlon_mismatch_file = "../outputs/latlon_mismatch.log"
stacode_mismatch_file = "../outputs/sta_mismatch.log"

# modules
def convert_to_datetime(eve_origin, source):

    if source == "wf_lst":
        return datetime.strptime(eve_origin, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            microsecond=0
        )
        # return datetime.strptime(eve_origin, "%Y/%m/%d %H:%M:%S.%f").replace(
        #     microsecond=0
        # )
    elif source == "st_lst":
        return datetime.strptime(str(eve_origin), "%Y%m%d")


def avg_vs(vs):
    w_kayen = 0.8
    w_array = 0.6
    w_pwave = 0.4 if vs["PWAVE_VS"] > 700.0 else 0.2
    w_hv = 0.4
    w_usgs = 0.2
    w_asscm = 0.2

    vs_vals = np.array(
        [
            vs["ASSCM_VS"],
            vs["KAYEN_VS"],
            vs["USGS_VS"],
            vs["HVSR_VS"],
            vs["PWAVE_VS"],
            vs["ARRAY_VS"],
        ]
    )
    vs_weights = np.array([w_asscm, w_kayen, w_usgs, w_hv, w_pwave, w_array])

    # Mask to identify non-NaN values
    valid_mask = ~np.isnan(vs_vals) & ~np.isnan(vs_weights)

    # Apply mask to values and weights
    valid_values = vs_vals[valid_mask]
    valid_weights = vs_weights[valid_mask]
    if not list(valid_values):
        weighted_vs = np.nan
    else:
        weighted_vs = np.average(valid_values, weights=valid_weights)

    return weighted_vs


def get_band_code(sampling_rate, corner_period):
    # Define the data
    data = {
        "Band code": ["F", "G", "D", "C", "E", "S", "H", "B"],
        "Band type": [
            "...",
            "...",
            "...",
            "...",
            "Extremely Short Period",
            "Short Period",
            "High Broad Band",
            "Broad Band",
        ],
        "Sample rate range (Hz)": [
            "≥ 1000 to < 5000",
            "≥ 1000 to < 5000",
            "≥ 250 to < 1000",
            "≥ 250 to < 1000",
            "≥ 80 to < 250",
            "≥ 10 to < 80",
            "≥ 80 to < 250",
            "≥ 10 to < 80",
        ],
        "Corner period (sec)": [
            "≥ 10 sec",
            "< 10 sec",
            "< 10 sec",
            "≥ 10 sec",
            "< 10 sec",
            "< 10 sec",
            "≥ 10 sec",
            "≥ 10 sec",
        ],
    }

    # Create the pandas DataFrame
    df = pd.DataFrame(data)

    # Function to parse sample rate range
    def parse_sample_rate_range(sample_rate_range):
        # Remove '≥ ' and '< ' and split the range at 'to'
        min_rate, max_rate = (
            sample_rate_range.replace("≥ ", "").replace("< ", "").split(" to ")
        )
        return float(min_rate), float(max_rate)

    # Apply the function to the 'Sample rate range (Hz)' column
    df[["min_sample_rate", "max_sample_rate"]] = (
        df["Sample rate range (Hz)"].apply(parse_sample_rate_range).apply(pd.Series)
    )

    # Function to parse corner period
    def parse_corner_period(corner_period):
        # Remove '≥ ' and '< ' to focus on the number and convert it to float
        # Remove the ' sec' unit from the string before converting it to float
        if "<" in corner_period:
            min_per, max_per = 0.0, 10.0
        else:
            min_per, max_per = 10.0, 100000.0
        return min_per, max_per

    # Add a column for the numeric corner period
    df[["min_corner_period", "max_corner_period"]] = (
        df["Corner period (sec)"].apply(parse_corner_period).apply(pd.Series)
    )

    # Find the row that matches the sampling rate and corner period
    matching_row = df[
        (df["min_sample_rate"] <= sampling_rate)
        & (df["max_sample_rate"] > sampling_rate)
        & (df["min_corner_period"] <= corner_period)
        & (df["max_corner_period"] > corner_period)
    ]
    # Return the band code if a match is found, otherwise return None
    if not matching_row.empty:
        return matching_row["Band code"].values[0]
    else:
        return "NOPE"


def get_cha_code(pazfile, sampling_rate, original_channel_code):

    # compute corner frequency
    tr = Trace()
    sacpz.attach_paz(tr, pazfile, torad=True)
    resp = inventory.response.Response.from_paz(
        tr.stats.paz["zeros"], tr.stats.paz["poles"], 1.0
    )
    min_freq = 0.01
    t_samp = 1.0 / sampling_rate
    nfft = int(sampling_rate / min_freq)
    cpx_response, freq = resp.get_evalresp_response(t_samp=t_samp, nfft=nfft)
    mag = abs(cpx_response)
    threshold = 0.7 * max(mag)
    # Find the index of the frequency where the amplitude drops below the threshold
    corner_freq_idx = np.where(mag > threshold)[0][0]
    corner_frequency = freq[corner_freq_idx]

    if corner_frequency == 0:
        corner_period = 10.0
    elif corner_frequency > 0.0:
        corner_period = 1 / corner_frequency
    # get the band code for the given corner frequency and sampling rate
    band_code = get_band_code(sampling_rate, corner_period)
    # assign the revised channel code based on the computed band code!
    revised_channel_code = "".join((band_code, original_channel_code[1::]))

    return corner_frequency, revised_channel_code


# read waveform, vs, and station csv files
wf_lst = pd.read_csv(wf_lst_file)
sta_lst = pd.read_csv(sta_lst_file, sep="\t", usecols=range(14))
vs_lst = pd.read_csv(sta_vs_file)

# filter NaN PAZ entries in station csv file!
sta_lst = sta_lst[sta_lst["PAZFILE"].notnull()]
sta_lst = sta_lst.reset_index()

# add datetime to wf_lst and sta_lst
wf_lst["event_origin"] = wf_lst["ORIGIN"].apply(convert_to_datetime, source="wf_lst")
sta_lst["sta_start"] = sta_lst["START"].apply(convert_to_datetime, source="st_lst")
sta_lst["sta_stop"] = sta_lst["STOP"].apply(convert_to_datetime, source="st_lst")

# add constant to the sta_lst
sta_lst["CONSTANT"] = sta_lst["SEN"] * sta_lst["GAIN"] * sta_lst["RECSEN"]

# add [weighted]average vs to the vs_lst
vs_lst["avg_vs"] = vs_lst.apply(avg_vs, axis=1)

# # replace Nan location with ''
# wf_lst["LOC"] = wf_lst["LOC"].fillna("")
# wf_lst["LOC"] = wf_lst["LOC"].apply(
#     lambda x: f"{int(x):02d}" if isinstance(x, (float, int)) else ""
# )

# read anu_II_IU_S1 inventory
# inv_anu_ii_iu_s1 = read_inventory(inv_ANU_II_IU_S1_file)
inv_anu_ii_iu_s1 = read_inventory(inv_AUSPASS_II_IU_file)

# read AU iventory
inv_au = read_inventory(inv_AU_file)

ASSCM_SC = []
KAYEN_SC = []
ASSCM_VS = []
KAYEN_VS = []
USGS_VS = []
HVSR_VS = []
PWAVE_VS = []
ARRAY_VS = []
vs_value = []
vs_flag = []
rev_net_code = []
paz_file = []
paz_crn_freq = []
paz_constant = []
rev_cha_code = []
st_elevation = []
st_description = []
df_net_log = pd.DataFrame()
# net_log = []
net_log = {
    "missing_network": [],
    "school_network": [],
    "inconsistent_network": [],
    "notunique_network": [],
    "missing_metadata": [],
}
paz_log = []


for index, row in wf_lst.iterrows():
    # add vs value, if available
    # region
    df_dum = (
        pd.merge(
            pd.DataFrame(row).transpose(),
            vs_lst[vs_lst["STA"] == row["STA"]],
            how="cross",
        )
        .assign(
            Distance=lambda r: r.apply(
                lambda x: gps2dist_azimuth(x["STLA"], x["STLO"], x["LAT"], x["LON"])[0],
                axis=1,
            )
        )
        .sort_values("Distance")
        .reset_index(drop=True)
    )
    # breakpoint();
    if len(df_dum):
        ASSCM_SC.append(
            df_dum["ASSCM_SC"][0] if df_dum["Distance"][0] < 100.0 else "Distance>100.0!"
        )
        KAYEN_SC.append(
            df_dum["KAYEN_SC"][0] if df_dum["Distance"][0] < 100.0 else "Distance>100.0!"
        )
        ASSCM_VS.append(
            df_dum["ASSCM_VS"][0] if df_dum["Distance"][0] < 100.0 else "Distance>100.0!"
        )
        KAYEN_VS.append(
            df_dum["KAYEN_VS"][0] if df_dum["Distance"][0] < 100.0 else "Distance>100.0!"
        )
        USGS_VS.append(
            df_dum["USGS_VS"][0] if df_dum["Distance"][0] < 100.0 else "Distance>100.0!"
        )
        HVSR_VS.append(
            df_dum["HVSR_VS"][0] if df_dum["Distance"][0] < 100.0 else "Distance>100.0!"
        )
        PWAVE_VS.append(
            df_dum["PWAVE_VS"][0] if df_dum["Distance"][0] < 100.0 else "Distance>100.0!"
        )
        ARRAY_VS.append(
            df_dum["ARRAY_VS"][0] if df_dum["Distance"][0] < 100.0 else "Distance>100.0!"
        )
        vs_value.append(
            df_dum["avg_vs"][0] if df_dum["Distance"][0] < 100.0 else "Distance>100.0!"
        )
        vs_flag.append("proxy" if np.isnan(df_dum["KAYEN_VS"][0]) else "measured")
    else:
        vs_value.append(np.nan)
        vs_flag.append("NA")
    
    # endregion

    # add revised network code attribute!
    # region
    row["REV_NET"] = row["NET"]
    code_tmp = sta_lst[sta_lst["STA"] == row["STA"]]["NETID"].unique()

    # case 1: if network code is missing
    if not isinstance(row["NET"], str):
        # check station code absent in `stationlist.dat`
        if code_tmp.size == 0:
            print("Error: No station found in the stationlist file!")
            break
        # check station code linked to multiple network codes!
        if code_tmp.size > 1:
            print("Error: More than one network for one station!")
            break
        # add network code to the station
        row["REV_NET"] = code_tmp[0]
        log_txt = "missing network code (%s) added for station %s!" % (
            code_tmp[0],
            row["STA"],
        )
        print(log_txt)
        net_log["missing_network"].append(log_txt)

    # case 2: if network code is "S"
    if row["NET"] == "S":
        row["REV_NET"] = "S1"
        log_txt = "Network code revised for %s station from S to S1" % row["STA"]
        print(log_txt)
        net_log["school_network"].append(log_txt)

    # case 3: if network code is not consistent with the one from station list!
    if not inv_anu_ii_iu_s1.select(
        network=row["REV_NET"], station=row["STA"]
    ):  # station is not part of inv_anu_ii
        if code_tmp.size == 1:
            if not row["NET"] == code_tmp[0]:
                row["REV_NET"] = code_tmp[0]
                log_txt = "network code revised for %s station from %s to %s" % (
                    row["STA"],
                    row["NET"],
                    row["REV_NET"],
                )
                print(log_txt)
                net_log["inconsistent_network"].append(log_txt)
        elif code_tmp.size == 0:
            log_txt = "%s station is not listed in stationlist file!" % row["STA"]
            print(log_txt)
            net_log["missing_metadata"].append(log_txt)
        else:
            log_txt = "%s station is linked to %d network!" % (
                row["STA"],
                code_tmp.size,
            )
            print(log_txt)
            net_log["notunique_network"].append(log_txt)

    rev_net_code.append(row["REV_NET"])
 
    # endregion

    # add corresponding paz file!
    # region
    if inv_anu_ii_iu_s1.select(
        network=row["REV_NET"],
        station=row["STA"],
        channel=row["CHA"],
        sampling_rate=row["SAMPLING_RATE"],
        time=row["event_origin"],
    ):
        paz_file.append("dataless")
        paz_crn_freq.append("dataless")
        paz_constant.append("dataless")
        rev_cha_code.append(row["CHA"])
    else:
        mask = (
            (sta_lst["NETID"] == row["REV_NET"])
            & (sta_lst["STA"] == row["STA"])
            & (sta_lst["sta_start"] <= row["event_origin"])
            & (sta_lst["sta_stop"] >= row["event_origin"])
            & (sta_lst["COMPONENT"] == row["CHA"])
        )
        if sta_lst["PAZFILE"][mask].size > 1:
            if (sta_lst["PAZFILE"][mask].unique().size == 1) & (
                sta_lst["CONSTANT"][mask].unique().size == 1
            ):
                # get paz file
                pazf = sta_lst["PAZFILE"][mask].unique()[0]
                pazf = "/".join(("../inputs/paz", pazf))
                pazf = pazf.rstrip()
                paz_file.append(pazf)
                paz_constant.append(sta_lst["CONSTANT"][mask].unique()[0])

                # assign the revised channel code
                corner_frequency, revised_channel_code = get_cha_code(
                    pazf, row["SAMPLING_RATE"], row["CHA"]
                )
                paz_crn_freq.append(corner_frequency)
                rev_cha_code.append(revised_channel_code)

                # add log text!
                log_txt = (
                    "Redundant information for %s in stationlist file!"
                    % ".".join((row["REV_NET"], row["STA"], row["CHA"]))
                )
                print(log_txt)
                paz_log.append(log_txt)
                mask = []
            else:
                print(
                    "ERROR: more than one PAZ file or Constant for %s in stationlist file!"
                    % ".".join((row["REV_NET"], row["STA"], row["CHA"]))
                )
                break
        elif sta_lst["PAZFILE"][mask].size == 1:
            # get paz file
            pazf = sta_lst["PAZFILE"][mask].values[0]
            pazf = "/".join(("../inputs/paz", pazf))
            pazf = pazf.rstrip()
            paz_file.append(pazf)
            paz_constant.append(sta_lst["CONSTANT"][mask].values[0])
            # assign the revised channel code

            corner_frequency, revised_channel_code = get_cha_code(
                pazf, row["SAMPLING_RATE"], row["CHA"]
            )
            paz_crn_freq.append(corner_frequency)
            rev_cha_code.append(revised_channel_code)

            mask = []
        else:
            paz_file.append("Missing!")
            paz_crn_freq.append("Missing!")
            paz_constant.append("Missing!")
            rev_cha_code.append(row["CHA"])
            mask = []
    # endregion

    # add station description and elevation from inventory files!
    # region
    if len(inv_anu_ii_iu_s1.select(network=row["REV_NET"], station=row["STA"])) > 0:
        tmp_des = inv_anu_ii_iu_s1.select(network=row["REV_NET"], station=row["STA"])
        description = tmp_des.get_contents()["stations"][0]
        elevation = tmp_des[0][0].elevation

    elif len(inv_au.select(network=row["REV_NET"], station=row["STA"])) > 0:
        tmp_des = inv_au.select(network=row["REV_NET"], station=row["STA"])
        description = tmp_des.get_contents()["stations"][0]
        elevation = tmp_des[0][0].elevation
    else:
        description = ""
        elevation = np.nan

    st_description.append(description)
    st_elevation.append(elevation)

    # endregion
    

wf_lst["KAYEN_SC"] = KAYEN_SC
wf_lst["ASSCM_SC"] = ASSCM_SC
wf_lst["KAYEN_VS"] = KAYEN_VS
wf_lst["ASSCM_VS"] = ASSCM_VS
wf_lst["USGS_VS"] = USGS_VS
wf_lst["PWAVE_VS"] = PWAVE_VS
wf_lst["HVSR_VS"] = HVSR_VS
wf_lst["ARRAY_VS"] = ARRAY_VS

wf_lst["VS"] = vs_value
wf_lst["VS_FLAG"] = vs_flag
wf_lst["REV_NET"] = rev_net_code
wf_lst["PAZ_FILE"] = paz_file
wf_lst["PAZ_CRN_FREQ"] = paz_crn_freq
wf_lst["PAZ_CONSTANT"] = paz_constant
wf_lst["REV_CHA"] = rev_cha_code
wf_lst["STEL"] = st_elevation
wf_lst["Description"] = st_description

df = wf_lst
df.to_csv(mod_wf_lst_file)


# TASK: list waveforms with possible channel code issues!
df.reset_index()
df = df[
    df.duplicated(
        subset=["STA", "CHA", "SAMPLING_RATE", "ORIGIN", "LOC"],
        keep=False,
    )
]
df = df[~df["CHA"].str[0].isin(["L", "V"])]  # remove L and V channels!
df.to_csv(report_duplicated_channels)

# TASK: network patching log

max_length = max(len(v) for v in net_log.values())

# Create a standardized dictionary by filling shorter lists with None
standardized_data = {}
for key, value in net_log.items():
    # Fill shorter lists with None to match the max length
    if len(value) < max_length:
        value = value + [None] * (max_length - len(value))
    standardized_data[key] = value

# Convert the standardized dictionary to a pandas DataFrame
df_net_log = pd.DataFrame(standardized_data)
df_net_log.to_csv(report_network_log)

# TASK: identify AU stations, same station, same channel, different location code
# (this cause ambiguity getting response from TA stationlist as it does not have loc code!)
grouped = wf_lst[wf_lst["REV_NET"] == "AU"].groupby(["STA", "CHA"])["LOC"].nunique()
grouped[grouped > 1].to_csv(report_au_sta_with_multi_loc_code)


# TASK: channel code changes!
wf_lst[wf_lst["CHA"] != wf_lst["REV_CHA"]].to_csv(report_cha_rev)

# TASK paz: file mapping to channel code in our database!
df_mapping = wf_lst.filter(["PAZ_FILE", "SAMPLING_RATE", "PAZ_CRN_FREQ", "REV_CHA"])
df_mapping.drop_duplicates()
df_mapping = df_mapping[
    df_mapping["PAZ_CRN_FREQ"].apply(lambda x: isinstance(x, (int, float)))
]
df_mapping.to_csv(paz_map_file)

# TASK: station VS log
df_vs_log = wf_lst.filter(["STA", "STLA", "STLO", "VS", "VS_FLAG"])
df_vs_log.to_csv(report_au_sta_vs)

# TASK: identify records with same network, same station, different lat or long!

# Group by 'STA' and 'NET' and find unique LAT or LON within each group
different_lat_lon = wf_lst.groupby(["STA", "NET"]).filter(
    lambda x: x["STLA"].nunique() > 1 or x["STLO"].nunique() > 1
)
different_lat_lon = different_lat_lon.drop_duplicates(subset=["STLA", "STLO"])
different_lat_lon.to_csv(latlon_mismatch_file)

# TASK: identify records that the station code from miniseed file is different for the one
# provided in the original wf_lst_file by Trev! look at WaveforAugmenter.py
df_sta_mismatch = wf_lst[wf_lst["STA"] != wf_lst["STA_ORIG"]]
df_sta_mismatch.to_csv(stacode_mismatch_file)
