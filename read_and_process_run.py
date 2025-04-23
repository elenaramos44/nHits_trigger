import sys
import os

# Añade el directorio padre al sys.path
sys.path.append(os.path.abspath("/eos/home-d/dcostasr/SWAN_projects/2025_data"))
sys.path.append(os.path.abspath("/eos/home-d/dcostasr/SWAN_projects/NiCf/offline_trigger"))

from src.read_data import process_and_write_parts, load_concatenated, read_mcc_offsets
from wcte.brbtools import sort_run_files, get_part_files

run = input("Which run do you want to process? (####) --> ")

print("Reading run and part files...")
run_files  = sort_run_files(f"/eos/experiment/wcte/data/2025_commissioning/offline_data/{run}/WCTE_offline_R{run}S*P*.root")
part_files = get_part_files(run_files)
mcc_map = read_mcc_offsets()

print(f"Run {run} has {len(part_files)} part files. You can read a single part, multiple, or all")
print(f"    Example_1: 3 (Single run file)")
print(f"    Example_2: 0,1,2,3,4 (List of runs)")
print(f"    Example_3: all (Read all runs. This might take a while.)")

parts_to_process = input("Which parts do you want to read? --> ")

# Case: "all"
if parts_to_process.strip().lower() == "all":
    process_and_write_parts(run_files, part_files, mcc_map, max_card=132, max_chan=19, outdir=f"tmp_parquet/{run}/")

# Case: input just one number, e.g. "3"
elif parts_to_process.isdigit():
    part_idx = int(parts_to_process)
    process_and_write_parts(run_files, part_files[part_idx], mcc_map, max_card=132, max_chan=19, outdir=f"tmp_parquet/{run}/")

# Case: comma separated list of numbers, e.g. "1,2,5"
else:
    try:
        parts_list = [int(p.strip()) for p in parts_to_process.split(',')]
        
        # Duplicates verification
        duplicates = [x for x in set(parts_list) if parts_list.count(x) > 1]
        if duplicates:
            raise ValueError(f"Duplicated values found in the list: {duplicates}")

        process_and_write_parts(run_files, [part_files[i] for i in parts_list], mcc_map, max_card=132, max_chan=19, outdir=f"tmp_parquet/{run}/")
        
    except ValueError as e:
        raise ValueError(f"Invalid input. {e}")
