#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Dump and evaluate predictions.
'''


import sys
import csv
from collections import Counter

from ..util.util import smart_open


def handle_predictions(conf, data, summary=(sys.stdout,), predict=()):
    '''
    Write predictions to TSV files and/or print evaluation summaries.
    '''
    evaluator = Evaluator(conf, predict)
    evaluator.evaluate(data)

    for file in summary:
        evaluator.summary(file)
    evaluator.dump_predictions()


class SummaryWriter:
    '''
    Write a summary line for each occurrence of a mention.
    '''

    fields = (
        'DOC_ID',
        'START',      # document character offset
        'END',        # ditto
        'MENTION',    # text
        'REF_ID',     # 1 or more
        'PRED_ID',
        'CORRECT',    # prediction matches reference
        'N_IDS',      # number of IDs for the top-ranked candidate name
        'REACHABLE',  # reference ID is among the candidates
    )

    def __init__(self, conf):
        self.fn = conf.logging.prediction_fn
        self._entries = []

    def update(self, mention, refs, occs, _y, _r, decision):
        '''Update with outcome information per occurrence.'''
        mention = _rm_tabs_nl(mention)
        for occ in occs:
            entry = (*occ, mention, refs, *decision)
            self._entries.append(entry)

    def dump(self):
        '''Sort and serialise all entries.'''
        # Put in corpus order (easier comparison across runs).
        self._entries.sort()
        with smart_open(self.fn, 'w') as f:
            writer = csv.writer(f, quotechar=None,
                                delimiter='\t', lineterminator='\n')
            writer.writerow(self.fields)  # add a header line
            writer.writerows(self._entries)


class DetailedWriter:
    '''
    Write a list of scored candidates for each distinct mention.
    '''
    def __init__(self, conf):
        self.fn = conf.logging.detailed_fn
        self._entries = {'correct': [], 'reachable': [], 'unreachable': []}

    def update(self, mention, refs, occs, _, ranking, outcome):
        '''Update with a distinct mention.'''
        category, pred = self._outcome(outcome)
        entry = (str(refs), _rm_tabs_nl(mention), occs, pred, ranking)
        self._entries[category].append(entry)

    def dump(self):
        '''Write to disk.'''
        for category in self._entries:
            with smart_open(self.fn.format(category), 'w') as f:
                writer = csv.writer(f, quotechar=None,
                                    delimiter='\t', lineterminator='\n')
                writer.writerows(self._iterentries(self._entries[category]))

    def _iterentries(self, entries):
        entries.sort()
        for refs, mention, occs, pred, ranking in entries:
            yield ('Concept:', refs)
            yield ('Term:', mention)
            yield ('Occs:', self._occ_summary(occs))
            yield ('Pred:', pred)
            yield ('Ranking:',)
            for s, c, i in ranking:
                s = '  {:.3f}'.format(s)
                i = ', '.join(i)
                yield s, c, i
            yield ()

    @staticmethod
    def _occ_summary(occs):
        docs = Counter(doc for doc, _, _ in occs)
        return ', '.join('{} ({})'.format(*i) for i in docs.most_common())

    @staticmethod
    def _outcome(outcome):
        id_, correct, _, reachable = outcome
        if correct:
            category = 'correct'
        elif reachable:
            category = 'reachable'
        else:
            category = 'unreachable'
        return category, id_


class TRECWriter:
    '''
    Write two tables for TREC evaluation.

    Prediction file format:
        qid, 0, docno, 0, sim, 0

    Gold file format:
        qid, 0, docno, label
    '''

    def __init__(self, conf):
        self.fn = conf.logging.trec_eval_fn
        self._entries = {'prediction': [], 'gold': []}

    def update(self, _mention, _refs, occs, label, ranking, _outcome):
        '''Add a sequence of 6 and 4 elements respectively'''
        for occ in occs:
            qid = '{}-{}-{}'.format(*occ)
            for (score, _, _), correct in zip(ranking, label):
                docno = len(self._entries['gold'])
                entry_prediction = (qid, 0, docno, 0, score, 0)
                entry_gold = (qid, 0, docno, int(correct))
                self._entries['prediction'].append(entry_prediction)
                self._entries['gold'].append(entry_gold)

    def dump(self):
        '''Write to disk.'''
        for category in self._entries:
            with smart_open(self.fn.format(category), 'w') as f:
                writer = csv.writer(f, quotechar=None,
                                    delimiter='\t', lineterminator='\n')
                writer.writerows(self._entries[category])


def _rm_tabs_nl(text):
    return text.replace('\t', ' ').replace('\n', ' ')


# Map command-line args to Writer instances.
_writer_names = {
    'summary': SummaryWriter,
    'rich': DetailedWriter,
    'trec': TRECWriter,
}


class Evaluator:
    '''
    Count a selection of outcomes and compute accuracy.
    '''
    def __init__(self, conf, writers=()):
        self.conf = conf
        self.writers = [_writer_names[w](conf) for w in writers]

        self.correct = 0
        self.total = 0
        self.unreachable = 0
        self.ambiguous = 0
        self.nocandidates = 0

    @property
    def accuracy(self):
        '''Proportion of mentions with correct top-ranked ID.'''
        return self.correct/self.total

    @classmethod
    def from_data(cls, conf, data, writers=()):
        '''
        Create an already populated Evaluator instance.
        '''
        evaluator = cls(conf, writers)
        evaluator.evaluate(data)
        return evaluator

    def evaluate(self, data):
        '''
        Evaluate all predictions and push entries to the writers.
        '''
        for scores, ids, cands, y, mention, refs, occs in self._iterranges(data):
            ranking = sorted(zip(scores, cands, ids), reverse=True)
            decision = self._decide(ranking, ids, refs)
            for writer in self.writers:
                writer.update(mention, refs, occs, y, ranking, decision)
            for _ in occs:
                self._update(*decision[1:])

    @staticmethod
    def _iterranges(data):
        for start, end, *annotation in data.mentions:
            scores = data.scores[start:end, 0]
            ids = data.ids[start:end]
            cands = data.candidates[start:end]
            labels = data.y[start:end, 0]
            yield (scores, ids, cands, labels, *annotation)

    def _decide(self, scored, ids, refs):
        id_ = self._disambiguate(scored)
        correct = id_ in refs
        n_ids = len(scored) and len(scored[0][-1])
        reachable = any(id_ in refs for id_ in set().union(*ids))
        return id_, correct, n_ids, reachable

    def _disambiguate(self, scored):
        try:
            score, _, top_ids = scored[0]
        except IndexError:
            # No candidates.
            return self.conf.general.nil_symbol

        if not top_ids:
            # Best-ranked candidate name has no match in the dictionary.
            return self.conf.general.nil_symbol

        if score < self.conf.rank.min_score:
            # Score too low.
            return self.conf.general.nil_symbol

        if len(top_ids) == 1:
            # Unambiguous case.
            return next(iter(top_ids))

        # Look for cues in the lower-ranked candidates.
        for _, _, ids in scored[1:]:
            common = top_ids.intersection(ids)
            if common:
                # Another name of the top-ranked concept(s) was among the
                # candidates. Remove the concepts to which it doesn't map.
                top_ids = common
                if len(top_ids) == 1:
                    break

        # If still ambiguous, pick the lowest-ordering ID to be deterministic.
        return min(top_ids)

    def _update(self, correct, n_ids, reachable):
        '''Update counts.'''
        self.total += 1
        self.correct += correct
        self.unreachable += not reachable
        if n_ids == 0:
            self.nocandidates += 1
        elif n_ids > 1:
            self.ambiguous += 1

    def summary(self, outfile):
        '''Write an evaluation summary to outfile.'''
        labels = 'accuracy correct total unreachable nocandidates ambiguous'
        for label in labels.split():
            outfile.write('{:12} {:5}\n'.format(label, getattr(self, label)))

    def dump_predictions(self):
        '''Write all accumulated predictions to disk.'''
        for writer in self.writers:
            writer.dump()
