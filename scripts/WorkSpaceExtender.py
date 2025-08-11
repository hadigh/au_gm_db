from gmprocess.io.asdf.stream_workspace import StreamWorkspace
from gmprocess.metrics.waveform_metric_collection import WaveformMetricCollection

from gmprocess.metrics.station_metric_collection import StationMetricCollection
from obspy import Stream
import matplotlib.pyplot as plt
from openquake.hazardlib.valid import gsim
from openquake.hazardlib.source import PointSource
from openquake.hazardlib.mfd import ArbitraryMFD
from openquake.hazardlib.scalerel import WC1994
from openquake.hazardlib.geo import Point, NodalPlane, geodetic
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.contexts import ContextMaker
import numpy as np
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth
import re
from pathlib import Path
import shutil
import argparse

"""
Python script that extends the workspace files of the processed events (with computed wm and sm) by storing the followings under Auxiliary

1- predictions by 15 selected ground motion models (see line 168 below) at all of the periods avialble for each of the GMMs; (the mean values are stored! exp(log(pre)) in g)

2- normalized observed SAs at periods same as those from GMM; the SA values are normalized to be on HR site condiciton with Vs760 m/s, and the normalization factor is computed 
for the selected gmm the values are obs_norm in g (it is not log)

3- GMM-distance for the event

the user can specify --event flag to only process one event otherwisse it would loop through all events in the gmprocess data directory (this required the wf_lst_f file list all
the events in the data directory!) 

Note- we should check if there is inconsistency between station metadata and computed metrics in case of having only Z component and so not GMrotd! line 150 and 151!

Parameters

"""


def gmm_pre(eve, sta_meta_data, gmm_id, hard_rock):

    if hard_rock:
        sta_vs30 = 760.0
    elif np.isnan(sta_meta_data["vs"]):
        sta_vs30 = 600.0
    else:
        sta_vs30 = sta_meta_data["vs"]

    sta_lat = sta_meta_data["latitude"]
    sta_lon = sta_meta_data["longitude"]
    sta_loc = Point(sta_lon, sta_lat)

    site_collection = SiteCollection(
        [
            Site(
                location=sta_loc,
                vs30=sta_vs30,
                vs30measured=False,
                z1pt0=40.0,
                z2pt5=1.0,
            )
        ]
    )

    # this is the source for which we compute the median ground shaking
    src_id = eve.id
    src_loc = Point(eve.longitude, eve.latitude)
    src_depth = eve.depth_km
    src_mag = eve.magnitude
    src = PointSource(
        source_id=src_id,
        name="point",
        tectonic_region_type="Active Shallow Crust",
        mfd=ArbitraryMFD(
            magnitudes=[src_mag],
            occurrence_rates=[1.0],
        ),
        rupture_mesh_spacing=2.0,
        magnitude_scaling_relationship=WC1994(),
        rupture_aspect_ratio=1.0,
        temporal_occurrence_model=PoissonTOM(50.0),
        upper_seismogenic_depth=0.0,
        lower_seismogenic_depth=50.0,
        location=src_loc,
        nodal_plane_distribution=PMF([(1.0, NodalPlane(strike=45, dip=50, rake=0))]),
        hypocenter_distribution=PMF([(1, src_depth)]),  # depth in km
    )
    ruptures = [r for r in src.iter_ruptures()]  # create list of ruptures

    ruptures[0].mag = (
        src_mag  # this is to make sure the magnitude is consistent and avoid python precision!
    )
    ruptures = [ruptures[0]]
    mags = [r.mag for r in ruptures]

    # gmm
    # instantiate the selected gsim
    if gmm_id == "NGAEastGMPE":
        gmm = [
            # gsim("[NGAEastGMPE]\ngmpe_table = 'NGA-East_Backbone_Model.geometric.hdf5'") # swap to this!
            gsim("[NGAEastGMPE]\ngmpe_table = 'NGAEast_PEER_GP.hdf5'")
        ]
    elif gmm_id == "TromansEtAl2019":
        gmm = [gsim("[TromansEtAl2019]\ngmpe_name = 'BooreEtAl2014'")]
    else:
        gmm = [gsim(gmm_id)]

    # gmm specific periods:
    # these are the intensity measure type for which we compute the median ground shaking
    idx = [index for index, element in enumerate(dir(gmm[0])) if "COEFF" in element][0]
    tmp1 = getattr(gmm[0], dir(gmm[0])[idx])
    tmp2 = getattr(tmp1, "sa_coeffs")
    gmm_periods = [str(element) for element in list(tmp2.keys())]
    imtls = {s: [0] for s in gmm_periods}

    # context_maker = ContextMaker('*', gmm, {'imtls': imtls})  # necessary contexts builder
    context_maker = ContextMaker(
        "*", gmm, {"imtls": imtls, "mags": ["%1.2f" % m for m in mags]}
    )

    ctxs = context_maker.from_srcs([src], site_collection)

    # add rrup that is used in gmm prediction
    sta_meta_data["rrup"] = ctxs[0].rrup[0]

    if ctxs:
        gms = context_maker.get_mean_stds(
            ctxs
        )  # calculate ground motions and stds, returns array of shape (4, G, M, N)
    else:
        gms = np.nan

    return gmm_periods, gms


def normalize_observation(obs, per_obs, log_pre, log_pre_hr, per_pre):
    # compute observation at the same period of predictions
    interp_obs = np.interp(per_pre, per_obs, obs)
    log_obs = np.log(interp_obs)
    log_obs_norm = log_obs + log_pre_hr.flatten() - log_pre.flatten()

    return log_obs_norm, log_obs


def distance_gmm_pre(eve, gmm_id):
    # # Ref Sites and Source
    src_loc = Point(0.0, 0.0)
    src_depth = eve.depth_km
    src_mag = eve.magnitude

    num_sites = 600
    points = geodetic.point_at(
        src_loc.longitude, src_loc.latitude, 0.0, np.logspace(-1, 3.0, num_sites)
    )
    sites_list = []
    for i in np.arange(num_sites):
        site = Site(
            Point(points[0][i], points[1][i]),
            vs30=760,
            z1pt0=40,
            z2pt5=1.0,
            vs30measured=True,
        )
        sites_list.append(site)
    sitec = SiteCollection(sites_list)

    src = PointSource(
        source_id="Ref",
        name="point",
        tectonic_region_type="Active Shallow Crust",
        mfd=ArbitraryMFD(
            magnitudes=[src_mag],
            occurrence_rates=[1.0],
        ),
        rupture_mesh_spacing=2.0,
        magnitude_scaling_relationship=WC1994(),
        rupture_aspect_ratio=1.0,
        temporal_occurrence_model=PoissonTOM(50.0),
        upper_seismogenic_depth=0.0,
        lower_seismogenic_depth=50.0,
        location=src_loc,
        nodal_plane_distribution=PMF([(1.0, NodalPlane(strike=45, dip=50, rake=0))]),
        hypocenter_distribution=PMF([(1, src_depth)]),  # depth in km
    )

    ruptures = [r for r in src.iter_ruptures()]  # create list of ruptures

    ruptures[0].mag = (
        src_mag  # this is to make sure the magnitude is consistent and avoid python precision!
    )
    ruptures = [ruptures[0]]
    mags = [r.mag for r in ruptures]

    # gmm
    # instantiate the selected gsim
    if gmm_id == "NGAEastGMPE":
        gmm = [
            # gsim("[NGAEastGMPE]\ngmpe_table = 'NGA-East_Backbone_Model.geometric.hdf5'")
            gsim("[NGAEastGMPE]\ngmpe_table = 'NGAEast_PEER_GP.hdf5'")
        ]
    elif gmm_id == "TromansEtAl2019":
        gmm = [gsim("[TromansEtAl2019]\ngmpe_name = 'BooreEtAl2014'")]
    else:
        gmm = [gsim(gmm_id)]

    # gmm specific periods:
    # these are the intensity measure type for which we compute the median ground shaking
    idx = [index for index, element in enumerate(dir(gmm[0])) if "COEFF" in element][0]
    tmp1 = getattr(gmm[0], dir(gmm[0])[idx])
    tmp2 = getattr(tmp1, "sa_coeffs")
    gmm_periods = [str(element) for element in list(tmp2.keys())]
    imtls = {s: [0] for s in gmm_periods}

    # context_maker = ContextMaker('*', gmm, {'imtls': imtls})  # necessary contexts builder
    context_maker = ContextMaker(
        "*", gmm, {"imtls": imtls, "mags": ["%1.2f" % m for m in mags]}
    )

    ctxs = context_maker.from_srcs([src], sitec)

    rrup = ctxs[0]["rrup"]

    if ctxs:
        gms = context_maker.get_mean_stds(
            ctxs
        )  # calculate ground motions and stds, returns array of shape (4, G, M, N)
    else:
        gms = np.nan

    return rrup, gmm_periods, gms


# Inputs
wf_lst_f = "../outputs/merged_au_wf_lst.csv"
gmprocess_projects_dir = "/home/pcuser/Projects/au_gm_db/gmprocess_projects/data"  # List all directories in a specific path
path = Path(gmprocess_projects_dir)

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--event", type=str, help="event id")

# Parse the arguments
args = parser.parse_args()

if args.event:
    directories = [path / args.event]
else:
    directories = [d for d in path.iterdir() if d.is_dir()]


# Main Code:
# read wf_lst file (getting vs30 from the list for each of the records!)
wf_lst = pd.read_csv(wf_lst_f)

for d in directories:
    f_d = d / "workspace.h5"
    f = str(f_d)

    # read workspace file of the event
    ws = StreamWorkspace.open(f)
    eve = ws.get_event(ws.get_event_ids()[0])
    # inv = ws.get_inventory()
    # sta_codes = ws.get_stations()

    # get "passed" stream metadata and metrics from ws!
    wmc = WaveformMetricCollection.from_workspace(ws, label="default")
    stsmd = wmc.stream_metadata  # STreamS Meta Data
    stsmt = wmc.waveform_metrics  # STreamS MeTrics

    # # get station metrics (incliudes Rrup to be saved under Aux!)
    # smc = StationMetricCollection.from_workspace(ws)

    # get the observed SA values, and sta metadata (add vs30 too to the metadata!)
    # store it in a dictionary named sta_obs

    sta_obs = {}
    sa_rotd_dfs = []
    for i, wml in enumerate(stsmt):
        # wf Metrics
        tmp_df = wml.select("PSA").to_df()
        tmp_df = tmp_df.loc[tmp_df["IMC"] == "RotD(percentile=50.0)"]
        sa_rotd_dfs.append(tmp_df)

        # Vs
        vs_sta = wf_lst[
            (wf_lst["event_origin"] == eve.origins[0].time)
            & (wf_lst["REV_NET"] == stsmd[i][0]["network"])
            & (wf_lst["STA"] == stsmd[i][0]["station"])
        ]["VS"].values[0]
        stsmd[i][0]["vs"] = float(vs_sta)

        # Station metrics (Rrup)
        # idx = [
        #     index
        #     for index, entry in enumerate(smc.stream_paths)
        #     if entry.startswith(".".join((stsmd[i][0]["network"], stsmd[i][0]["station"])))
        # ][0]
        # rrup_sta = smc.station_metrics[idx].rrup_mean
        # stsmd[i][0]["rrup"] = rrup_sta

    # check if an error needs to be raised! (when number of stations (sta_codes above) is not consistent with number of sa!!!)

    gmms_ids = [
        "Allen2012",
        "AtkinsonBoore2006",
        "BooreEtAl2014",
        "ChiouYoungs2008SWISS06",
        "ZhaoEtAl2006AscSWISS08",
        "ChiouYoungs2014",
        "AbrahamsonEtAl2014",
        "CampbellBozorgnia2014",
        "SomervilleEtAl2009YilgarnCraton",
        "SomervilleEtAl2009NonCratonic",
        "DrouetBrazil2015",
        "RietbrockEdwards2019Mean",
        "ESHM20Craton",
        "NGAEastGMPE",
        "ShahjoueiPezeshk2016",
    ]
    # gmms_ids = ["NGAEastGMPE"]
    # delete exisitng one (make it smarter later!)
    try:
        del ws.dataset._auxiliary_data_group["GMM"]
    except KeyError:
        pass

    data_type = "GMM"

    for gmm_id in gmms_ids:

        print(gmm_id)

        # perhaps I should not add predictions for Mag<4.0 for all GMMs! here just for sending more WS to abhi!
        if (eve.magnitude < 4.0) & (
            gmm_id == "NGAEastGMPE"
        ):  # perhaps I should not add predictions for Mag<4.0 for all GMMs!
            continue
        # exclude events with depth larger than 40 km (prehaps I should merge this with above condition!)
        if eve.depth_km > 40.0:
            continue

        #exclude events that have no passed streams!
        if not wmc.waveform_metrics:
            print(f"Skipping event_id {eve.id} as there is no passed stream!")
            continue

        # predictions at recording stations!
        for i in np.arange(len(sa_rotd_dfs)):

            # compute prediction only if sa_rotd is available for the station
            if not (sa_rotd_dfs[i].values.size == 0):

                sta_code = ".".join((stsmd[i][0]["network"], stsmd[i][0]["station"]))

                try:
                    gmm_period, gmm_prediction = gmm_pre(
                        eve, stsmd[i][0], gmm_id, hard_rock=False
                    )

                    # model predictions on hard rock site with vs30 of 760 m/s!
                    gmm_period, HR_gmm_prediction = gmm_pre(
                        eve, stsmd[i][0], gmm_id, hard_rock=True
                    )

                    data_per = [float(s[3:-1]) for s in gmm_period]
                except (
                    IndexError
                ):  # distance is larger than 10000 km so OQ does not calculate!
                    gmm_prediction = np.nan

                # # add to auxiliary if not nan!
                if not np.isnan(gmm_prediction).all():

                    # store predicted values for (GMM, sta)
                    # data = np.exp(gmm_prediction[0][0])
                    data = gmm_prediction
                    path = f"{gmm_id}/{sta_code}"
                    ws.dataset.add_auxiliary_data(
                        data=data,
                        data_type=data_type,
                        path=path,
                        parameters={
                            "Rrup": stsmd[i][0]["rrup"],
                            "Vs30": stsmd[i][0]["vs"],
                        },
                    )

                    # store (log of) normalized observed values for (sta) normalized using GMM
                    # get observed SA in unit of g
                    obs_sa = (
                        sa_rotd_dfs[i].Result.values / 100.0
                    )  # convert from %g to g!
                    obs_per = [
                        float(re.search(r"T=([\d.]+)", item).group(1))
                        for item in sa_rotd_dfs[i].IMT.values
                    ]
                    obs_norm, obs = normalize_observation(
                        obs_sa,
                        obs_per,
                        gmm_prediction[0][0],
                        HR_gmm_prediction[0][0],
                        data_per,
                    )

                    path = f"{gmm_id}/{'_'.join((sta_code, 'normalized-observation'))}"
                    ws.dataset.add_auxiliary_data(
                        data=obs_norm,
                        data_type=data_type,
                        path=path,
                        parameters={
                            "Rrup": stsmd[i][0]["rrup"],
                            "Vs30": stsmd[i][0]["vs"],
                            "Obs": obs,
                        },
                    )

        # predictions at GMM periods for distance from 0-600 km and hard rock site condition!
        Rrup, gmm_per, gmm_gms = distance_gmm_pre(eve, gmm_id)

        df = pd.DataFrame(
            np.hstack((gmm_gms[0][0], gmm_gms[1][0])),
            columns=[f"Rrup({value:.6f})" for value in Rrup]
            + [f"sigma_Rrup({value:.6f})" for value in Rrup],
            index=gmm_per,
        )

        # store distance predictions for all of periods in the auxiliary!
        path = f"{gmm_id}/distance"
        ws.dataset.add_auxiliary_data(
            data=np.array(df),
            data_type=data_type,
            path=path,
            parameters={
                "columns": [f"Rrup({value:.6f})" for value in Rrup]
                + [f"sigma_Rrup({value:.6f})" for value in Rrup],
                "rows": gmm_per,
            },
        )

        path_per = f"{gmm_id}/period"
        ws.dataset.add_auxiliary_data(
            data=np.array(data_per),
            data_type=data_type,
            path=path_per,
            parameters={},
        )

        print(ws.dataset.auxiliary_data)

    ws.close()

    directory_name = f_d.parent.name
    new_file_name = f"workspace_{directory_name}.h5"
    new_file_path = d / new_file_name  # Target location (current directory)

    # Copy the file to the new location with the modified name
    shutil.copy(f_d, new_file_path)
