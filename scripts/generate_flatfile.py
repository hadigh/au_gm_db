import os
from gmprocess.io.asdf.stream_workspace import StreamWorkspace
import pandas as pd
from obspy.geodetics import locations2degrees, gps2dist_azimuth



"""
Generate a CSV-format flatfile listing key parameters of interest.

Note:
- Currently includes only epicentral distance (retrieved from 'fit_spectra' in trace parameters).
- Ground motion parameters (e.g., PGA, PGV, SA) are not yet included.
- Future versions may incorporate additional source-to-site measures and engineering parameters.
"""


# Define the base directory
data_directory = "../gmprocess_projects/data"
output_file = "../outputs/flatfile.csv"
failed_output_file = "../outputs/failed_flatfile.csv"
# df data initiation
recs_ids = []
recs_hpcs = []
recs_dists = []
eqs_ids = []
eqs_mags = []
eqs_deps = []

# df_fail data initiation
fail_recs_ids = []
fail_recs_dists = []
fail_recs_reason = []
fail_eqs_ids = []
fail_eqs_mags = []
fail_eqs_deps = []

# Iterate through all directories within the data directory
for root, dirs, files in os.walk(data_directory):
    
    for directory in dirs:
        workspace_path = os.path.join(root, directory, "workspace.h5")

        # Check if workspace.h5 exists in the directory
        if os.path.exists(workspace_path):
            print(f"Processing: {workspace_path}")

            # Open the workspace using gmprocess
            try:
                workspace = StreamWorkspace.open(workspace_path)

                # List streams with the 'default' tag (processed)
                event_id = workspace.get_event_ids()[0]
                eve = workspace.get_event(event_id)
                streams = workspace.get_streams(event_id, labels=["default"])
                print(f"Event ID: {event_id}")
                for stream in streams:
                    if stream.passed:
                        for tr in stream:
                            eqs_ids.append(event_id)
                            eqs_mags.append(eve.magnitude)
                            eqs_deps.append(eve.depth_km)        
                            recs_ids.append(tr.id)
                            recs_dists.append(tr.get_parameter('fit_spectra')['epi_dist']) # change this later?
                            recs_hpcs.append(tr.get_parameter('corner_frequencies')['highpass'])
                    else:
                        # get epicentral distance for the stream
                        sta_lat = stream[0].stats.coordinates.latitude
                        sta_lon = stream[0].stats.coordinates.longitude
                        distance_m, az12, az21 = gps2dist_azimuth(eve.latitude, eve.longitude,
                                                                   sta_lat, sta_lon) 
                        # Convert meters to kilometers
                        distance_km = distance_m / 1000
                        
                        for tr in stream:
                            fail_recs_ids.append(tr.id)
                            fail_recs_dists.append(distance_km)
                            fail_eqs_ids.append(event_id)
                            fail_eqs_mags.append(eve.magnitude)
                            fail_eqs_deps.append(eve.depth_km)
                            if tr.passed:
                                msg = 'Passed'
                            else:
                                msg = tr.get_parameter('failure')['reason']
                            fail_recs_reason.append(msg) 

            except Exception as e:
                print(f"Error processing {workspace_path}: {e}")

            # Close the workspace
            workspace.close()

# Create a dictionary from the lists. The keys will be the column names.
data = {
    'earthquake_id': eqs_ids,
    'record_id': recs_ids,
    'hpc': recs_hpcs,
    'distance': recs_dists,
    'magnitude': eqs_mags,
    'depth': eqs_deps
}

fail_data = {
    'earthquake_id': fail_eqs_ids,
    'record_id': fail_recs_ids,
    'fail_reason': fail_recs_reason,
    'distance': fail_recs_dists,
    'magnitude': fail_eqs_mags,
    'depth': fail_eqs_deps
}

# Create the DataFrame from the dictionary
df = pd.DataFrame(data)

# Create the DataFrame from the dictionary for failed records!
df_fail = pd.DataFrame(fail_data)

# Print the DataFrame to see the result
df.to_csv(output_file)
            
# Print the DataFrame to see the result
df_fail.to_csv(failed_output_file)
                  

          