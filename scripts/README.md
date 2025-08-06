# to write scripts!
0- check my revisions 1,2, and 3 first!
1- run WaveformFileVerifier.py to check if all the waveforms do exist!
1.5 - [optional] run PazFileVerifier to check which pazfile listed in stationlist.dat is not avaialable in ../inputs/paz folder!
2- run WaveformListAugmenter.py (change lines 27 and 32 if needs be!)
3- run WaveformListStationListStationVsMerger.py (change 55-63!)
4- run GMInputGenerator.py (change line 38; do not know why but had to run it in two segments one until end 2022 and second one from start 2023 till end (line 181))
5- run PortalStationxmlGenerator.py (change line 271, and 275)

6- after running gmrecords assemble, run  Assemly validator





# my Revisions! 😁️

1) 👋️ wrote fix_xhannel_codes.py to manually(!) assign the right channel codes to the traces within problematic streams and save the corrected miniseed file in wf_data folder!
(note in the future if I replace them from NAS, I need to re-do this!)

2) 👋️ after correspondance with Robert Pickle (anu), had to manually download the waveforms 2023-02-05T00.35.WG.WBP10.mseed, 2022-12-11T14.30.WG.WBP10.mseed, 2023-01-05T05.08.WG.WBP10.mseed
as the original ones had wrong sampling rate of 250 HZ for only vertical components! (note in the future if I replace them from NAS, I need to re-do this!)

3) 👋️ Manually edited the wf_lst.csv as follwing mseed files were misassigned!

BRAT was assigned to AUMAG the correct mseed was: 2023-06-29T15.28.S1.AUMAG.mseed

MBWA was assigned to NWAO  the correct mseed was: 2002-06-23T11.19.IU.NWAO.mseed

manually fixed the waveform path for followings:
TA: It seems as if the download period for MUN was fixed for several records – I have added the following files to the “new_hsd” folder and referenced these in the file “20250130_updated_au_wf_lst4ghd_mun_fix.csv”.  Note, IRIS these downloads come with HH* by default.
2022-01-22T07.55.AU.MUN.mseed
2022-01-24T21.22.AU.MUN.mseed
2022-01-24T21.48.AU.MUN.mseed
2022-02-01T10.39.AU.MUN.mseed

## 4 and 5 below fixed by Eric during the visit to USGS!
#4) in waveform_metric_calculator add:👋️ (path: /home/hadi/miniconda3/envs/gmprocess/lib/python3.9/site-packages/gmprocess/metrics/waveform_metric_calculator.py

       #### HG
        # print(self.steps.items())
        ss = self.stream
        if ss.num_horizontal < 2:
            filtered_dict = {
                key: value for key, value in self.steps.items() if "rotd" not in key
            }

        # metric is something like "channels-pga", i.e., an imc-imt
        # metric_steps is the list of operations that will produce the
        #   metric, such as "reduce.TraceMax"
        # for metric, metric_steps in self.steps.items():
        for metric, metric_steps in filtered_dict.items():
        

#5) in flatfile.py comment out like this:👋️ (path: /home/hadi/miniconda3/envs/gmprocess/lib/python3.9/site-packages/gmprocess/io/asdf/flatfile.py)



	    ## HG
            # with open(default_config_file, "r", encoding="utf-8") as f:
            #     yaml = YAML()
            #     yaml.preserve_quotes = True
            #     default_config = yaml.load(f)
            # update_dict(self.workspace.config, default_config)

       
6) in HumanReviewGUI.py add 👋️

if tr.passed:
in line 350!





#############################################################################################################
Note: adding lines to config.yml should be by space and not tab!!! (if tab you get strange errors)

unknown problem:
f = '../gmprocess_projects/data/20200415071104/raw/AU.CNB..BHE__2020-04-15T07:07:00.019538Z__2020-04-15T07:36:59.994538Z.mseed'

# Fail a stream if any of the constituent traces failed?
 any_trace_failures: False

- check_instrument:
        n_max: 3
        n_min: 1
        require_two_horiz: False 
  
- check_clipping:
        threshold: 1.0
        
check_instrument

snr_check



#5) 👋️ revised the "1P" network dataless seed file by running
#anu_response_reviser.py
#dev_inv_anu_tmp.py
#merge_ANU_II_IU_S1.py
#6) 👋️ Updated the channel codes for problematic stations, as identified by running WaveformListAugmenter.py. 
#These stations had channel code misassignments that prevented the streams from being merged. Applied the corrections using fix_channel_codes.py.
#Note, the followings are noise and are ignored!
#1994-08-06T11.05.MEL.PIN.mseed

####################################################ISSUES and HINTS #############################
- manually revised the strec config file to point to the right directory!
- some reeatation lines in stationlist.dat with UM network (WV)


