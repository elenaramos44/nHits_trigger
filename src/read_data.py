import glob
import os
import uproot

import awkward as ak
import numpy   as np

from tqdm import tqdm

def read_file(files, parts, evt_offset):
    tree = uproot.open(files[parts] + ":WCTEReadoutWindows")
    
    # Read the variables
    file_hit_card_ids    = ak.values_astype(tree["hit_mpmt_card_ids"]  .array(), np.int64)
    file_hit_channel_ids = ak.values_astype(tree["hit_pmt_channel_ids"].array(), np.int64)
    file_hit_times       = ak.values_astype(tree["hit_pmt_times"]      .array(), np.int64)
    file_hit_charges     = ak.values_astype(tree["hit_pmt_charges"]    .array(), np.int64)
    event_number         = ak.values_astype(tree["event_number"]       .array(), np.int64)

    # Number of hits per event
    n_hits_per_event = ak.num(file_hit_card_ids)

    # Create the repeated event array
    file_event_number = np.repeat(event_number, n_hits_per_event) + evt_offset
    file_event_number = ak.unflatten(file_event_number, n_hits_per_event)

    return file_hit_card_ids, file_hit_channel_ids, file_hit_times, file_hit_charges, file_event_number

def process_and_write_parts(run_files, good_parts, mcc_map, max_card, max_chan, outdir="tmp_parquet"):
    os.makedirs(outdir, exist_ok=True)
    evt_offset = 0

    for i, part in enumerate(tqdm(good_parts, desc="Procesando partes")):
        file_hit_card_ids, file_hit_channel_ids, file_hit_times, file_hit_charges, file_event_number = read_file(run_files, part, evt_offset)

        # Build lookup
        lookup = np.zeros((max_card + 1, max_chan + 1))
        for (card, chan), shift in mcc_map.items():
            lookup[card, chan] = shift

        # Hit Times Correction
        flat_card_ids     = ak.ravel(file_hit_card_ids)
        flat_chan_ids     = ak.ravel(file_hit_channel_ids)
        flat_corrections  = lookup[flat_card_ids, flat_chan_ids]
        corrections       = ak.unflatten(flat_corrections, ak.num(file_hit_card_ids))

        term1 = file_hit_times
        term2 = (file_event_number // 512) * (2**35)
        term3 = ((file_event_number % 512 == 511) & (file_hit_times < 2**34)) * (2**35)
        corrected_times = term1 + term2 + term3 + corrections

        # Save to Disk
        # Need Snappy Conversion For Swan Usage
        ak.to_parquet(file_hit_card_ids,     f"{outdir}/card_ids_part{i}.parquet")
        ak.to_parquet(file_hit_channel_ids,  f"{outdir}/channel_ids_part{i}.parquet")
        ak.to_parquet(file_hit_charges,      f"{outdir}/charges_part{i}.parquet")
        ak.to_parquet(file_event_number,     f"{outdir}/event_number_part{i}.parquet")
        ak.to_parquet(corrected_times,       f"{outdir}/hit_times_part{i}.parquet")

        evt_offset = file_event_number[-1][-1] + 1

def load_concatenated(outdir="./tmp_parquet"):
    def load(pattern):
        files = sorted(glob.glob(os.path.join(outdir, pattern)))
        return ak.concatenate([ak.from_parquet(f) for f in files], axis=0)

    return {
        "card_ids":        load("card_ids_part*.parquet"),
        "channel_ids":     load("channel_ids_part*.parquet"),
        "charges":         load("charges_part*.parquet"),
        "event_number":    load("event_number_part*.parquet"),
        "hit_times":       load("hit_times_part*.parquet"),
    }