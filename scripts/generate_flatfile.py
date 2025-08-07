import os
from gmprocess.io.asdf.stream_workspace import StreamWorkspace
import pandas as pd


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

# df data initiation
recs_ids = []
recs_hpcs = []
recs_dists = []
eqs_ids = []
eqs_mags = []
eqs_deps = []

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
                    # breakpoint()   

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

# Create the DataFrame from the dictionary
df = pd.DataFrame(data)

# Print the DataFrame to see the result
print(df)
df.to_csv(output_file)
            

            

          