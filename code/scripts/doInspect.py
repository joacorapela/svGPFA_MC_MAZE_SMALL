from pynwb import NWBHDF5IO

# Replace with the path to your NWB file
file_path = 'sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb'

with NWBHDF5IO(file_path, 'r') as io:
    nwbfile = io.read()

    # Print basic metadata
    print(f"Session Description: {nwbfile.session_description}")
    print(f"Identifier: {nwbfile.identifier}")
    print(f"Session Start Time: {nwbfile.session_start_time}")

    # List acquisition data
    print("\nAcquisition:")
    for name, data in nwbfile.acquisition.items():
        print(f" - {name}: {type(data)}")

    # List stimulus data
    print("\nStimulus:")
    for name, data in nwbfile.stimulus.items():
        print(f" - {name}: {type(data)}")

    # List processing modules
    print("\nProcessing Modules:")
    for name in nwbfile.processing:
        print(f" - {name}")

    # List units (spike data)
    if nwbfile.units is not None:
        print("\nUnits Table Columns:")
        print(nwbfile.units.colnames)

    # List intervals (e.g., trials)
    if nwbfile.intervals is not None:
        print("\nIntervals:")
        for name in nwbfile.intervals:
            print(f" - {name}")

