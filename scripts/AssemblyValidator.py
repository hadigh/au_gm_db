import os
from gmprocess.io.asdf.stream_workspace import StreamWorkspace
import pandas as pd

"""
Verifies that all MiniSEED files in the event's raw directory are stored under the unprocessed tag in the workspace file.
(i.e. detects cases where the response information is missing for any of the raw data files!)
for example, found this:
Event ID: 20001223071323
differences for event 20001223071323 are {'II.WRAB.00.BH', 'II.WRAB.10.BH'}
"""

# Define the base directory
data_directory = "../gmprocess_projects/data"

# store those with assembly issue!
output_log = "../outputs/gmprocess_assembly_issue.log"

# Iterate through all directories within the data directory
diff = {}
for root, dirs, files in os.walk(data_directory):
    for directory in dirs:
        workspace_path = os.path.join(root, directory, "workspace.h5")

        # Check if workspace.h5 exists in the directory
        if os.path.exists(workspace_path):
            print(f"Processing: {workspace_path}")

            # Open the workspace using gmprocess
            try:
                workspace = StreamWorkspace.open(workspace_path)

                # List streams with the 'unprocessed' tag
                event_ids = workspace.get_event_ids()
                streams_ids = []
                for event_id in event_ids:
                    streams = workspace.get_streams(event_id, labels=["unprocessed"])
                    print(f"Event ID: {event_id}")
                    for stream in streams:
                        streams_ids.append(stream.get_id())
            except Exception as e:
                print(f"Error processing {workspace_path}: {e}")

            # Close the workspace
            workspace.close()
            
            raw_directory = os.path.join(root, directory, "raw")
            if os.path.exists(raw_directory):
                mseed_files = [
                    f for f in os.listdir(raw_directory) if f.endswith(".mseed")
                ]

                data = []
                for mseed_file in mseed_files:
                    try:
                        network, station, location, channel = mseed_file.replace(
                            ".", "_"
                        ).split("_")[:4]
                        data.append(
                            {
                                "network": network,
                                "station": station,
                                "location": location,
                                "instrument": channel[0:-1],
                            }
                        )
                    except ValueError:
                        print(f"Skipping invalid file name: {mseed_file}")

                df = pd.DataFrame(
                    data, columns=["network", "station", "location", "instrument"]
                )
                df["ID"] = df["network"] + "." + df["station"] + "." + df["location"] + "." + df["instrument"]

                diff[event_id] = set(df["ID"]) - set(streams_ids)
                print(f"differences for event {event_id} are {diff[event_id]}")

# Clear the file at the start
open(output_log, "w").close()

with open(output_log, "w") as f:
    for event_id, differences in diff.items():
        if differences:  # only write if set is not empty
            f.write(f"Event ID: {event_id}\n")
            f.write(f"Differences for event {event_id} are {sorted(differences)}\n\n")

