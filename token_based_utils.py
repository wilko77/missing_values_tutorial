import pickle
import csv
from array import array
from itertools import repeat
from functools import reduce

from anonlink.similarities._utils import sort_similarities_inplace
from clkhash.comparators import NgramComparison, NonComparison


def tokenize_entities(filename, schema):
    """returns a list of the tokens for each entity"""
    tokenizers = [NonComparison()] + [field.hashing_properties.comparator for field in schema.fields[1:]]
    with open(filename, 'rt') as f:
        csvreader = csv.reader(f)
        csvreader.__next__()  # skip header
        entity_tokens = []
        for row in csvreader:
            tokens = []
            for i, (tokn, feat) in enumerate(zip(tokenizers, row)):
                tks = set(f'{i} {t}' for t in tokn.tokenize(feat))
                tokens.append(tks)
            entity_tokens.append(tokens)
        with open(f'{filename}.tkn.pkl', 'wb') as pf:
            pickle.dump(entity_tokens, pf)
        return entity_tokens


def dice_per_feature(toks_a, counts_a, toks_b, counts_b):
    """compute the dice coefficient on a per-feature basis.
    toks_x are lists of sets of token, counts_x are list of the corresponding set counts.
    """
    sim = 0
    f_count = 0
    for tok_a, count_a, tok_b, count_b in zip(toks_a, counts_a, toks_b, counts_b):
        if count_a > 0 and count_b > 0:
            sim += 2 * len(tok_a.intersection(tok_b)) / (count_a + count_b)
            f_count += 1
    if f_count > 0:
        num = len(counts_a) - f_count
        return (0.95 ** num) * sim / f_count
    else:
        return 0


def sim_fun(per_feature, datasets, threshold, k=None):
    n_datasets = len(datasets)
    if n_datasets < 2:
        raise ValueError(f'not enough datasets (expected 2, got {n_datasets})')
    elif n_datasets > 2:
        raise NotImplementedError(
            f'too many datasets (expected 2, got {n_datasets})')
    filters0, filters1 = datasets

    result_sims = array('d')
    result_indices0 = array('I')
    result_indices1 = array('I')

    if not filters0 or not filters1:
        # Empty result of the correct type.
        return result_sims, (result_indices0, result_indices1)

    if per_feature:
        f1_counts = tuple(tuple(len(f) for f in f1) for f1 in filters1)
    else:
        filters1 = [reduce(set.union, f1) for f1 in filters1]
        f1_counts = tuple(len(f1) for f1 in filters1)

    for i, f0 in enumerate(filters0):
        if per_feature:
            f0_count = tuple(len(f) for f in f0)
        else:
            f0 = reduce(set.union, f0)
            f0_count = len(f0)
        if f0_count:
            if per_feature:
                coeffs = (dice_per_feature(f0, f0_count, f1, f1_count) for f1, f1_count in zip(filters1, f1_counts))
            else:
                coeffs = (
                    2 * len(f0.intersection(f1)) / (f0_count + f1_count)
                    for f1, f1_count in zip(filters1, f1_counts))
        else:  # Avoid division by zero.
            coeffs = repeat(0., len(filters1))

        cands = filter(lambda c: c[1] >= threshold, enumerate(coeffs))
        sorted_cands = sorted(cands, key=lambda x: -x[1])

        result_sims.extend(sim for _, sim in sorted_cands)
        result_indices0.extend(repeat(i, len(sorted_cands)))
        result_indices1.extend(j for j, _ in sorted_cands)

    sort_similarities_inplace(result_sims, result_indices0, result_indices1)

    return result_sims, (result_indices0, result_indices1)