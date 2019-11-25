import operator
from itertools import groupby

import pandas as pd


def load_true_matches(ds1, ds2, id_col='id'):
    """ extract the true matches from the given datasets. Link on 'id_col'"""
    dfa = pd.read_csv(ds1)
    dfb = pd.read_csv(ds2)
    a = pd.DataFrame({'ida': dfa.index,
                      'link': dfa[id_col]})
    b = pd.DataFrame({'idb': dfb.index,
                      'link': dfb[id_col]})
    dfj = a.merge(b, on='link', how='inner').drop(columns=['link'])
    the_truth = set()
    for row in dfj.itertuples(index=False):
        the_truth.add((row[0], row[1]))
    return the_truth


def greedy_solve(candidates_pairs):
    """greedily resolves the candidate pairs into matches, such that each entity has not more than one partner"""
    sims, dset_is, rec_is = candidates_pairs
    if len(dset_is) != len(rec_is):
        raise ValueError('inconsistent shape of index arrays')
    if len(dset_is) != 2:
        raise NotImplementedError('only binary solving is supported')

    dset_is0, dset_is1 = dset_is
    rec_is0, rec_is1 = rec_is
    if not (len(sims)
            == len(dset_is0) == len(dset_is1)
            == len(rec_is0) == len(rec_is1)):
        raise ValueError('inconsistent shape of index arrays')

    matches0 = set()
    matches1 = set()
    for rec_i0, sim, rec_i1 in zip(rec_is0, sims, rec_is1):
        if rec_i0 not in matches0 and rec_i1 not in matches1:
            matches0.add(rec_i0)
            matches1.add(rec_i1)
            yield sim, (rec_i0, rec_i1)


def compute_accuracies(candidates_pairs, true_matches):
    """greedily solve candidates_pairs, and compare to true matches."""
    sims = []
    tps = []
    fps = []
    fns = []

    found_matches = set()
    possible_matches = greedy_solve(candidates_pairs)

    for sim, g in groupby(possible_matches, key=operator.itemgetter(0)):
        new_pairs = map(operator.itemgetter(1), g)
        found_matches.update(new_pairs)
        tp = len(found_matches & true_matches)
        fp = len(found_matches - true_matches)
        fn = len(true_matches - found_matches)
        sims.append(sim)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)

    precisions = [tp / (tp + fp) for tp, fp in zip(tps, fps)]
    recalls = [tp / (tp + fn) for tp, fn in zip(tps, fns)]
    return precisions, recalls
