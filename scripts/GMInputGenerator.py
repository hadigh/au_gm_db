import pandas as pd
import os, shutil
from strec.subtype import SubductionSelector
import json
from obspy import read, read_inventory
from obspy.io.sac import sacpz
from obspy.core.inventory import Inventory, Network, Station, Site, Channel
from obspy.core.inventory.response import Response
import scipy.signal as signal
import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy import UTCDateTime

"""
Python script that generates input files for usgs-gmprocess for earthquakes listed in merged_au_wf_lst.csv 
merged_au_wf_lst.csv file is the output of WaveformListStationListMerger.py and contains information on both
waveforms and recording station

Parameters

Inputs:

wf_lst_file: The file path for the merged waveform list with station list, it also specifies the source for 
the recording station metadata and if such information is missing! output of WaveformListStationListMerger.py

wf_dir: The directory path where the waveform files in MiniSEED format are stored.

gp_dir: path to the gmprocess projects directory

paz_dir: The directory path where the paz files are stored.

inv_ANU_II_IU_S1.xml: inventory that is compiled from dataless files available for listed stations! output of merge_ANU_II_IU_S1.py

Outputs:


"""
# Inputs

# wf_lst_file = "../outputs/merged_2021_WP_wf_lst.csv"  # this is just for one event!
wf_lst_file = "../outputs/merged_au_wf_lst.csv"
gp_dir = "../gmprocess_projects/data"
inv_ANU_II_IU_S1_file = "../inputs/inv_AUSPASS_II_IU.xml"

# Functions
def check_and_trim_trace(tr, eve_ot, eve_lon, eve_lat, sta_lon, sta_lat):
    """
    Check and trim a trace if longer than 1 hour. Trim window is centered on P-arrival time:
    20 minutes before and after if available.

    Parameters:
    tr : obspy.Trace
        The seismic trace to evaluate and possibly trim.
    eve_ot : UTCDateTime
        Earthquake origin time.
    eve_lon, eve_lat : float
        Earthquake longitude and latitude.
    sta_lon, sta_lat : float
        Station longitude and latitude.

    Returns:
    obspy.Trace
        The trimmed (or original) trace.
    """
    # Step 1: Calculate epicentral distance in degrees
    distance_deg = locations2degrees(eve_lat, eve_lon, sta_lat, sta_lon)

    # Step 2: Compute P-wave travel time using TauPyModel (IASP91)
    model = TauPyModel(model="iasp91")
    arrivals = model.get_travel_times(
        source_depth_in_km=10, distance_in_degree=distance_deg, phase_list=["P", "p"]
    )
    if not arrivals:
        breakpoint()
        print("No P-wave arrival found.")
        return tr

    # Get the first P-wave arrival
    travel_time = arrivals[0].time if hasattr(arrivals[0], "time") else None
    if travel_time is None:
        print("No valid travel time found.")
        return tr
    p_time = eve_ot + travel_time

    # Step 3: Check if trace is longer than 1 hour
    duration = tr.stats.endtime - tr.stats.starttime
    if duration <= 3600:
        return tr  # No trimming needed

    # Step 4: Trim 20 min before and after P, respecting available bounds
    pre_p = 20 * 60  # 20 minutes in seconds
    post_p = 20 * 60

    start_trim = max(tr.stats.starttime, p_time - pre_p)
    end_trim = min(tr.stats.endtime, p_time + post_p)

    if start_trim >= end_trim:
        # breakpoint()
        print(
            "Trim window invalid; returning 3-sec trace! would be rejected by gmprocess!"
        )
        print(f"p-time after trace end time! ignoring {e}:{tr.stats.station}")
        end_trim = start_trim + 3

    tr.trim(starttime=start_trim, endtime=end_trim)
    return tr


def read_pazfile(pazfile):

    paztxt = open(pazfile).readlines()

    # get zeros first
    zeros = []
    num = paztxt[0].split("\t")
    nzeros = int(num[1])
    for i in range(1, nzeros + 1):
        tmpz = paztxt[i].strip("\n").split("\t")

        zeros.append(
            complex(float(tmpz[0]), float(tmpz[1])) * 2 * np.pi
        )  # convert to angular frequency

    # now get poles
    poles = []
    num = paztxt[nzeros + 1].split("\t")
    npoles = int(num[1])
    for i in range(nzeros + 2, nzeros + npoles + 2):
        tmpp = paztxt[i].strip("\n").split("\t")

        poles.append(
            complex(float(tmpp[0]), float(tmpp[1])) * 2 * np.pi
        )  # convert to angular frequency

    # get constant
    constant = paztxt[nzeros + npoles + 2].split("\t")
    constant = float(constant[1])

    # get normalising frequency
    normf = paztxt[nzeros + npoles + 3].split("\t")
    normf = float(normf[1])

    return poles, zeros, constant, normf


def generate_response(total_gain, paz_file, instrument_type):

    if instrument_type == "V":
        input_units = "M/S"
    elif instrument_type == "A":
        input_units = "M/S**2"

    poles, zeros, constant, normf = read_pazfile(paz_file)
    angc = 2.0 * np.pi

    normf = 5.0  # maybe change this to 5 Hz to be safe

    # get amp at normf - updated 2019-01-04
    b, a = signal.zpk2tf(zeros, poles, 1.0)
    freq = np.arange(0.1, 50.0, 0.1)
    w, resp = signal.freqs(b, a, freq * angc)
    cal_resp = abs(np.exp(np.interp(np.log(normf), np.log(freq), np.log(abs(resp)))))
    constant = 1.0 / cal_resp

    resp = Response.from_paz(
        zeros,
        poles,
        total_gain,
        normf,
        input_units=input_units,
        output_units="COUNTS",
        normalization_frequency=normf,
        normalization_factor=constant,
        pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
    )

    return resp


def build_single_inv(tr_lat, tr_lon, tr):

    if tr.stats.channel[-1] == "Z":
        orientation = [0.0, -90.0]
    elif (tr.stats.channel[-1] == "N") | (tr.stats.channel[-1] == "1"):
        orientation = [0.0, 0.0]
    elif (tr.stats.channel[-1] == "E") | (tr.stats.channel[-1] == "2"):
        orientation = [90.0, 0.0]

    inv_tmp = Inventory()

    net = Network(code=tr.stats.network, stations=[])

    sta = Station(
        code=tr.stats.station,
        latitude=tr_lat,
        longitude=tr_lon,
        elevation=345.0,
        site=Site(name=tr.stats.station),
    )

    cha = Channel(
        code=tr.stats.channel,
        location_code=tr.stats.location,
        latitude=tr_lat,
        longitude=tr_lon,
        elevation=345.0,
        depth=10.0,
        azimuth=orientation[0],
        dip=orientation[1],
        sample_rate=tr.stats.sampling_rate,
    )

    # Now tie it all together.
    cha.response = tr.stats.response
    sta.channels.append(cha)
    net.stations.append(sta)
    inv_tmp.networks.append(net)
    return inv_tmp


# read anu_II_IU_S1 inventory
inv_anu_ii_iu_s1 = read_inventory(inv_ANU_II_IU_S1_file)

# read waveform list
wf_lst = pd.read_csv(wf_lst_file)

# drop waveform traces with "Missing!" metadata!
wf_lst = wf_lst[wf_lst["PAZ_FILE"] != "Missing!"]
wf_lst = wf_lst.reset_index()

# drop traces with epicentral distance > max_dist km, and magnitude < min_mag
max_dist = 1500.0  # km
min_mag = 3.75 #MW
wf_lst = wf_lst[(wf_lst["REPI"] <= max_dist) & (wf_lst["MW"] >= min_mag)]
wf_lst = wf_lst.reset_index(drop=True)

# generate gmprocess input files
eve_ids = wf_lst.event_origin.unique()
# breakpoint()
for e in eve_ids:

    # check if event directory exists, if so DELETE it and create a new one; if not,
    # create one with raw folder as sub-directory!
    eve_id = e.replace(":", "").replace(" ", "").replace("-", "")

    if os.path.exists("/".join((gp_dir, eve_id))):
        shutil.rmtree("/".join((gp_dir, eve_id)))
        os.makedirs("/".join((gp_dir, eve_id, "raw")))
    else:
        os.makedirs("/".join((gp_dir, eve_id, "raw")))

    idx_event_data = wf_lst["event_origin"] == e

    # create event json file (report if there is a missing event parameter!) and \
    # create strec_result.jason file
    try:
        eve_dict = {
            "id": eve_id,
            "time": e,
            "latitude": float(wf_lst.EQLA[idx_event_data].values[0]),
            "longitude": float(wf_lst.EQLO[idx_event_data].values[0]),
            "depth_km": float(wf_lst.EQDEP[idx_event_data].values[0]),
            "magnitude": float(wf_lst.MW[idx_event_data].values[0]),
            "magnitude_type": "mw",
        }
        with open("/".join((gp_dir, eve_id, "event.json")), "w") as ef:
            ef.write(json.dumps(eve_dict))

        selector = SubductionSelector()
        strec_dict = selector.getSubductionType(
            eve_dict["latitude"],
            eve_dict["longitude"],
            eve_dict["depth_km"],
            eve_dict["magnitude"],
            eve_dict["id"],
        ).to_dict()
        with open("/".join((gp_dir, eve_id, "strec_results.json")), "w") as f:
            json.dump(strec_dict, f)

    except ValueError:
        print("NOTE- Event %s has missing parameters, REMOVED!!!!" % e)
        shutil.rmtree("/".join((gp_dir, eve_id)))
        continue

    # read raw data, revise net and channel codes, and merge traces, if required
    for f in wf_lst.mseed_path[idx_event_data].unique():
        st = read(f)

        # Merge
        try:
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
                print("Problematic Stream!: %s" % st)
                st.merge(method=-1)

            else:
                # apply default merge!
                st.merge(method=0, fill_value="interpolate")

        except:
            print("NOTE: could not merge data for %s" % st)

        # I- revise net and channel codes for each trace in the stream (if metadata exist for the trace!)
        # then write the mseed file for each trace
        # II- create Inventory for each stream and then write stationxml

        inv = Inventory()
        for tr in st:
            idx_tr = (wf_lst["mseed_path"] == f) & (wf_lst["CHA"] == tr.stats.channel)

            if idx_tr.any():
                tr.stats.network = wf_lst[idx_tr]["REV_NET"].values[0]
                tr.stats.channel = wf_lst[idx_tr]["REV_CHA"].values[0]

                # write trace to gmprocess input directory!
                tr_file_name = (
                    tr.id
                    + "__"
                    + str(tr.stats.starttime)
                    + "__"
                    + str(tr.stats.endtime)
                    + ".mseed"
                )
                tr = check_and_trim_trace(
                    tr,
                    UTCDateTime(wf_lst[idx_tr]["ORIGIN"].iloc[0]),
                    wf_lst[idx_tr]["EQLO"].iloc[0],
                    wf_lst[idx_tr]["EQLA"].iloc[0],
                    wf_lst[idx_tr]["STLO"].iloc[0],
                    wf_lst[idx_tr]["STLA"].iloc[0],
                )

                # tr = check_trace_mseed_quality(tr)

                tr.write(
                    "/".join((gp_dir, eve_id, "raw", tr_file_name)), format="MSEED"
                )

                tr_paz_file = wf_lst[idx_tr]["PAZ_FILE"].values[0]
                if tr_paz_file == "dataless":

                    tr_inv = inv_anu_ii_iu_s1.select(
                        network=tr.stats.network,
                        station=tr.stats.station,
                        channel=tr.stats.channel,
                        sampling_rate=tr.stats.sampling_rate,
                        time=e,
                    )
                else:

                    tr_total_gain = float(wf_lst[idx_tr]["PAZ_CONSTANT"].values[0])
                    normf = 5.0  # check this!
                    if tr.stats.channel[1] == "N":
                        instrument_type = "A"
                        input_units = "M/S**2"
                    else:
                        instrument_type = "V"
                        input_units = "M/S"
                    # sacpz.attach_paz(tr, tr_paz_file, torad=True)
                    # resp = Response.from_paz(
                    #     tr.stats.paz["zeros"],
                    #     tr.stats.paz["poles"],
                    #     tr_total_gain,
                    #     normf,
                    #     input_units=input_units,
                    #     output_units="VOLTS",
                    #     normalization_frequency=normf,
                    #     pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
                    # )
                    resp = generate_response(
                        tr_total_gain, tr_paz_file, instrument_type
                    )
                    tr.stats.response = resp
                    tr_lat = float(wf_lst[idx_tr]["STLA"].values[0])
                    tr_lon = float(wf_lst[idx_tr]["STLO"].values[0])
                    tr_inv = build_single_inv(tr_lat, tr_lon, tr)

                inv = inv + tr_inv
        # write the inventory for each stream
        # inv_file_name = ".".join((tr.stats.network, tr.stats.station, "xml")) #cause issues where you have response info for one instrument and not for the other one in colocated streams!
        inv_file_name = ".".join(
            (inv.networks[0].code, inv.networks[0].stations[0].code, "xml")
        )
        inv.write(
            "/".join((gp_dir, eve_id, "raw", inv_file_name)),
            format="stationxml",
            validate=True,
        )
