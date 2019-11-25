import base64
import os
import pickle

import anonlink
from bitarray import bitarray
from clkhash import clk

from eval_utils import compute_accuracies

SECRET = ",./,./"


def deserialize_bitarray(bytes_data):
    ba = bitarray(endian='big')
    data_as_bytes = base64.decodebytes(bytes_data.encode())
    ba.frombytes(data_as_bytes)
    return ba


def deserialize_filters(filters):
    res = []
    for i, f in enumerate(filters):
        ba = deserialize_bitarray(f)
        res.append(ba)
    return res


def generate_clks(filename, schema):
    with open(filename, 'rt') as f:
        hashed_data = clk.generate_clk_from_csv(f, SECRET, schema)
        return deserialize_filters(hashed_data)


def pr_curve_from_clks(clks_a, clks_b, threshold, true_matches):
    results_candidate_pairs = anonlink.candidate_generation.find_candidate_pairs(
            [clks_a, clks_b],
            anonlink.similarities.dice_coefficient,
            threshold
    )
    precisions, recalls = compute_accuracies(results_candidate_pairs, true_matches)

    return precisions, recalls


def run_series(folder, schema_dict, true_matches):
    for label, schema in schema_dict.items():
        print(f'current schema: {label}')
        for i in range(0, 81, 5):
            print(f'working on {i}% missing values')
            ds1 = os.path.join(folder, f'{i:02}_A.csv')
            ds2 = os.path.join(folder, f'{i:02}_B.csv')
            clks_a = generate_clks(ds1, schema)
            clks_b = generate_clks(ds2, schema)
            precisions, recalls = pr_curve_from_clks(clks_a, clks_b, 0, true_matches)
            res_file = os.path.join(folder, f'{i:02}_pr_{label}.pkl')
            with open(res_file, 'wb') as f:
                pickle.dump((precisions, recalls), f)
            print('done')
