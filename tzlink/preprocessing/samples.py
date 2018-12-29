#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Create training samples from input documents.
'''


import io
import os
import pickle
import logging
import tarfile
import multiprocessing as mp
from collections import defaultdict, namedtuple
from copy import deepcopy

import numpy as np

from ..datasets.load import load_data, load_dict
from .vectorize import load_wemb, Vectorizer
from .overlap import TokenOverlap
from ..candidates.generate_candidates import candidate_generator
from ..util.util import CacheDict, TypeHider, smart_open, ConfHash
from .elmo import elmo_default


class Sampler:
    '''
    Central unit for creating training samples.

    Includes all preprocessing steps.
    Holds references to embedding vocab and matrix,
    terminology index, candidate generator, etc.
    '''
    def __init__(self, conf):
        self.conf = conf

        # All attributes are loaded lazily.
        self._terminology = None
        self.emb = CacheDict(self._load_embeddings)
        self._cand_gen = None
        self._pool = None

    @property
    def terminology(self):
        '''Indexed terminology.'''
        if self._terminology is None:
            logging.info('loading terminology...')
            self._terminology = load_dict(self.conf)
        return self._terminology

    def _load_embeddings(self, which):
        econf = self.conf[which]
        logging.info('loading pretrained embeddings...')
        voc_index, emb_matrix = load_wemb(self.conf, econf)
        logging.info('loading vectorizer...')
        vectorizers = (Vectorizer(econf, voc_index, size)
                       for size in ('sample_size', 'context_size'))
        return EmbeddingInfo(*vectorizers, emb_matrix)

    @property
    def cand_gen(self):
        '''Candidate generator.'''
        if self._cand_gen is None:
            logging.info('loading candidate generator...')
            self._cand_gen = candidate_generator(self)
        return self._cand_gen

    @property
    def vectorizers(self):
        '''
        All vectorizers.

        For each zone (mention/context), there is a list of
        vectorizers, one for each embedding.
        '''
        embs = [self.emb[e] for e in self.conf.rank.embeddings]
        return {'mention': [e.vectorizer for e in embs],
                'context': [e.ctxt_vect for e in embs if e.ctxt_vect.length]}

    @property
    def pool(self):
        '''Multiprocessing pool for parallel sample generation.'''
        if self._pool is None:
            logging.info('initializing multiprocessing pool')
            self._pool = mp.Pool(self.conf.candidates.workers,
                                 initializer=_set_global_instances,
                                 initargs=[self.cand_gen, self.vectorizers])
        return self._pool

    def training_samples(self):
        '''Default-value wrapper around self.samples().'''
        subset = self.conf.general.training_subset
        oracle = self.conf.candidates.oracle['train']
        return self.samples(subset, oracle)

    def prediction_samples(self):
        '''Default-value wrapper around self.samples().'''
        subset = self.conf.general.prediction_subset
        oracle = self.conf.candidates.oracle['predict']
        return self.samples(subset, oracle)

    def samples(self, subset, oracle):
        '''
        Create vectorized samples with labels for training or prediction.
        '''
        logging.info('preprocessing dataset "%s"', subset)
        if not self.conf.general.dataset_cache:
            return self._samples(subset, oracle)

        cache_fn = self._cached_dataset_fn(subset, oracle)
        if os.path.exists(cache_fn):
            logging.info('loading cached dataset from %s', cache_fn)
            with smart_open(cache_fn, 'rb') as f:
                data = DataSet.load(f)
        else:
            data = self._samples(subset, oracle)
            logging.info('saving cached dataset at %s', cache_fn)
            with smart_open(cache_fn, 'wb') as f:
                data.save(f)
        return data

    def elmo_from_dict(self,dict,term,missing):
        try:
            return dict[term]
        except KeyError:
            missing.append(term)
            return term

    def _samples(self, subset, oracle):
        ranges = []
        weights = []
        samples = []  # holds 9-tuples of arrays
        if self.conf.emb_elmo.mention:
            logging.info('including elmo embedding...')
            potential_cache_terms = str(self.conf.emb_elmo.use_term_dict)
            try:
                '''
                1. read in cache dict, where key='term' and value=[elmo emb]
                2. for mentions and candidates that are already keys, use those
                3. for those that are not, save them in missing_terms and compute the elmo emb for those later
                '''
                import pickle
                elmo_term_vectorizer_dictionary = pickle.load(open(potential_cache_terms,'rb'))
                logging.info('saved elmo term embedding loaded: %s',potential_cache_terms)
                missing_terms = []
                for item, numbers in self._itercandidates(subset, oracle):
                    (mention, ref, context), occs = item
                    offset, length = len(samples), len(numbers)
                    ranges.append((offset, offset+length, mention, ref, occs))
                    mention_elmo = self.elmo_from_dict(elmo_term_vectorizer_dictionary,mention,missing_terms)
                    for candidate in numbers:
                        candidate.extend([self.elmo_from_dict(elmo_term_vectorizer_dictionary,candidate[7],missing_terms),mention_elmo])
                    samples.extend(numbers)
                    weights.extend(len(occs) for _ in range(length))
            except IOError: # no previously saved dict
                missing_terms = []
                for item, numbers in self._itercandidates(subset, oracle):
                    (mention, ref, context), occs = item
                    offset, length = len(samples), len(numbers)
                    ranges.append((offset, offset+length, mention, ref, occs))
                    missing_terms.append(mention)
                    for candidate in numbers:
                        missing_terms.append(candidate[7])
                        candidate.extend([candidate[7],mention])
                    samples.extend(numbers)
                    weights.extend(len(occs) for _ in range(length))    
            if len(missing_terms)!=0: # resolve missing terms
####produce elmo embedding, for (1) new dict, (2) supplementary dict
#import pdb; pdb.set_trace()
                logging.info('generating elmo term embedding...')
                missing_terms = list(set(missing_terms))
                t_b = self.conf.emb_elmo.term_processing_batch
                chunks = [missing_terms[x:x+t_b] for x in range(0, len(missing_terms), t_b)]
                term_elmo_emb = [c for chunk in elmo_default(chunks) for c in chunk]
                elmo_term_vectorizer_dictionary_missing = dict(zip(missing_terms, term_elmo_emb))
                try:
                    elmo_term_vectorizer_dictionary_updated = {**elmo_term_vectorizer_dictionary, **elmo_term_vectorizer_dictionary_missing}
                except NameError:
                    elmo_term_vectorizer_dictionary_updated = elmo_term_vectorizer_dictionary_missing
                if self.conf.emb_elmo.save_term:
                    import pickle
                    save_to = str(self.conf.emb_elmo.save_term_dict)
                    logging.info('updated dictionary saved to %s',save_to)
                    with open(save_to,'wb') as f:
                        pickle.dump(elmo_term_vectorizer_dictionary_updated,f)
                samples_missing = deepcopy(samples)
                samples = []
                for candidate_missing in samples_missing: # reshaping the data for DataSet class
                    candidate = deepcopy(candidate_missing)[:-2]
                    m = deepcopy(candidate_missing)[-1]
                    c = deepcopy(candidate_missing)[-2]
                    for i,t in zip([0,1],[m,c]):
                        if isinstance(t,str):
                            candidate[i].append(elmo_term_vectorizer_dictionary_updated[t])
                        else:
                            candidate[i].append(t)
                    samples.append(tuple(candidate))
            else: # no missing terms
                for candidate_missing in samples_missing: # reshaping the data for DataSet class
                    candidate = deepcopy(candidate_missing)[:-2]
                    for i,t in zip([0,1],[m,c]):
                        candidate[i].append(t)
                    samples.append(tuple(candidate))
        else: # elmo not used
            for item, numbers in self._itercandidates(subset, oracle):
                (mention, ref, _), occs = item
                offset, length = len(samples), len(numbers)
                ranges.append((offset, offset+length, mention, ref, occs))
                samples.extend(numbers)
                weights.extend(len(occs) for _ in range(length))
        import pdb; pdb.set_trace()
        data = DataSet(ranges, weights, *zip(*samples))
        logging.info('generated %d pair-wise samples (%d with duplicates)',
                     len(data.y), sum(data.weights))
        return data

    def _itercandidates(self, subset, oracle):
        logging.info('loading corpus...')
        corpus = load_data(self.conf, subset, self.terminology)
        mentions = _deduplicated(corpus)
        workers = self.conf.candidates.workers
        logging.info('generating candidates with %d workers...', workers)
        if workers >= 1:
            # Group into chunks and flatten the result.
            chunks = [(chunk, oracle) for chunk in self._chunks(mentions, 100)]
            vectorized = self.pool.imap(_worker_task, chunks)
            vectorized = (elem for chunk in vectorized for elem in chunk)
        else:
            vectorized = _task(mentions, oracle, self.cand_gen, self.vectorizers)
        yield from zip(mentions.items(), vectorized)

    @staticmethod
    def _chunks(items, size):
        chunk = []
        for item in items:
            chunk.append(item)
            if len(chunk) == size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _cached_dataset_fn(self, subset, oracle):
        confhash = self._confhash(subset, oracle)
        return self.conf.general.dataset_cache.format(confhash)

    def _confhash(self, subset, oracle):
        '''
        Assemble all relevant settings into a hash key.
        '''
        h = ConfHash()

        h.add(subset)
        h.add(oracle)
        h.add(self.conf.candidates.generator)
        h.add(self.conf.candidates.suppress_ambiguous)
        h.add(self.conf.rank.embeddings)

        ignored = {'rootpath', 'timestamp', 'workers', 'trainable'}
        for section in sorted(self.conf):
            if section.startswith('emb') or section == self.conf.general.dataset:
                for name in sorted(self.conf[section]):
                    if name not in ignored:
                        h.add(self.conf[section][name])

        # When changes to the code make obsolete previously cached datasets,
        # update this token.
        h.add('trim word-vector table to actual vocabulary')

        return h.hexdigest()


EmbeddingInfo = namedtuple('EmbeddingInfo', 'vectorizer ctxt_vect emb_matrix')


class DataSet:
    '''
    Container for original and vectorized input/output data.

    Attributes:
        x_q, x_a: vocabulary vectors of question and answer
            side (lists of 2D numpy arrays)
            If multiple embeddings are used, then each list
            contains multiple arrays.
        x_scores: confidence scores from candidate generation
            (1D or 2D numpy array)
        x_overlap: proportion of overlapping tokens
            (1D numpy array)
        x: the list [x_q_1, ... x_q_n, x_a_1, ... x_a_n, x_scores]
        y: the labels (2D numpy array)
        weights: counts of repeated samples (1D numpy array)
        scores: store the predictions here
        candidates: candidate name of each sample (list of str)
            Holds the same information as x_a, but using a
            human-readable plain-text version.
        ids: candidate IDs (list of set of str)
            A set of IDs for each sample. Most of the time,
            the set has one member only, except for the
            cases where a candidate name maps to multiple
            concepts.
        mentions: a list of mention-specific data for each
            candidate set. Each entry is a 5-tuple
                <i, j, mention, refs, occs>
            where i/j are the start/end indices wrt. to
            the rows of the sample vectors and occs is a
            list of <docid, start, end> triples identifying
            the occurrences in the corpus.
    '''
    def __init__(self, mentions, weights,
                 x_q, x_a, ctxt_q, ctxt_a, x_scores, x_overlap, y,
                 cands, ids):
        # Original data.
        self.mentions = mentions
        # Vectorized data.
        self.x_q = self._word_input_shape(x_q)
        self.x_a = self._word_input_shape(x_a)
        self.ctxt_q = self._word_input_shape(ctxt_q)
        self.ctxt_a = self._word_input_shape(ctxt_a)
        self.x_scores = np.array(x_scores)
        self.x_overlap = np.array(x_overlap)
        self.y = np.array(y)
        self.weights = np.array(weights)  # repetition counts
        self.scores = None  # hook for predictions
        # A set of candidate IDs for each sample.
        self.candidates = cands
        self.ids = ids

    @property
    def x(self):
        '''List of input tensors.'''
        return [*self.x_q, *self.x_a, *self.ctxt_q, *self.ctxt_a,
                self.x_scores, self.x_overlap]

    @staticmethod
    def _word_input_shape(vecs):
        return [np.array(v) for v in zip(*vecs)]

    def save(self, file):
        '''
        Export data to a gzipped tar file.
        '''
        with tarfile.open(fileobj=file, mode='w:gz') as tar:
            for name, bio in self._serialized_items():
                info = tarfile.TarInfo(name)
                info.size = bio.getbuffer().nbytes
                tar.addfile(info, bio)

    def _serialized_items(self):
        yield 'self', self._serialize(self, array=False)

        for name, value in self.__dict__.items():
            if self._picklable(value):
                continue
            if isinstance(value, list):
                for i, v in enumerate(value):
                    n = '{}.{}'.format(name, i)
                    yield n, self._serialize(v)
            else:
                yield name, self._serialize(value)

    @staticmethod
    def _serialize(value, array=True):
        bio = io.BytesIO()
        if array:
            np.save(bio, value)
        else:
            pickle.dump(value, bio)
        bio.seek(0)  # prepare for reading later
        return bio

    def __getstate__(self):
        '''
        Prevent pickling of numpy arrays.
        '''
        state = {name: value if self._picklable(value) else None
                 for name, value in self.__dict__.items()}
        return state

    @staticmethod
    def _picklable(obj):
        if isinstance(obj, np.ndarray):
            return False
        if isinstance(obj, list) and obj and isinstance(obj[0], np.ndarray):
            return False
        return True

    @classmethod
    def load(cls, file):
        '''
        Create a DataSet instance from a tar.gz created with save().
        '''
        with tarfile.open(fileobj=file) as tar:
            members = iter(tar)
            info = next(members)
            if info.name != 'self':
                raise ValueError(
                    'invalid DataSet dump: expected "self" as first member')
            with tar.extractfile(info) as f:
                dataset = pickle.load(f)

            for info in members:
                with tar.extractfile(info) as f:
                    # Workaround: tar.extractfile returns a BufferedReader
                    # object, which makes np.load think it is a "real" file.
                    # It tries to load the data using np.fromfile, which fails.
                    # The TypeHider wraps the file-like object in a different
                    # type, while still offering all (non-special) methods and
                    # attributes.
                    array = np.load(TypeHider(f))
                    cls._add_array(dataset.__dict__, info.name, array)
        return dataset

    @staticmethod
    def _add_array(attribs, name, array):
        try:
            name, i = name.split('.')
        except ValueError:
            attribs[name] = array
        else:
            if i == '0':
                attribs[name] = []
            attribs[name].append(array)


def _deduplicated(corpus):
    mentions = defaultdict(list)
    for doc in corpus:
        docid = doc['docid']
        for sec in doc['sections']:
            offset = sec['offset']
            context = sec['text']
            for mention in sec['mentions']:
                key = mention['text'], mention['id'], context
                occ = docid, offset + mention['start'], offset + mention['end']
                mentions[key].append(occ)
    return mentions


# Global variables are necessary to allow Pool workers to re-use the same
# instances across all tasks.
# https://stackoverflow.com/a/10118250
CAND_GEN = None
VECTORIZERS = None

def _set_global_instances(cand_gen, vectorizers):
    global CAND_GEN
    global VECTORIZERS
    CAND_GEN = cand_gen
    VECTORIZERS = vectorizers


def _worker_task(arg):
    return list(_task(*arg, CAND_GEN, VECTORIZERS))


def _task(items, oracle, cand_gen, vectorizers):
    '''
    Iterate over numeric representations of all samples.

    Returns:
        iter(list(tuple))
            Yield a list of samples for each mention.
            Each sample is a 9-list
            <x_q, x_a, ctxt_q, ctxt_a, scores, overlap, y, cand, ids>.
    '''
    overlap = TokenOverlap()
    def _vectorize(text, zone):
        return [v.vectorize(text) for v in vectorizers[zone]]

    for mention, ctxt_q, samples in cand_gen.samples_many(items, oracle):
        vec_q = _vectorize(mention, 'mention')
        ctxt_q = _vectorize(ctxt_q, 'context')
        data = []
        for cand, score, ctxt_a, label in samples:
            vec_a = _vectorize(cand, 'mention')
            ctxt_a = _vectorize(ctxt_a, 'context')
            data.append([
                vec_q,
                vec_a,
                ctxt_q,
                ctxt_a,
                score,
                overlap.overlap(mention, cand),
                (float(label),),
                cand,
                cand_gen.terminology.ids([cand]),
            ])
        yield data
