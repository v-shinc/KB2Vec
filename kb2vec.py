#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys
import os
import heapq
import time
from copy import deepcopy
import threading
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from numpy import exp, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, tile, diag,ones

logger = logging.getLogger("models.kb2vec")

# sys.path.append('D:\\Source Code\\gensim-develop')
import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange



try:
    from kb2vec_inner import train_triple_sg, FAST_VERSION
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1
    def train_triple_sg(model,triple,alpha,work=None,neur0=None,detR=None):
        e1,r,e2 = triple[0],triple[1],triple[2]

        if model.negative:

            labels = zeros(model.negative + 1).astype(REAL)
            labels[0] = 1.0
            l1 = deepcopy(model.syn0[e1.index])
            entity_indices = [e2.index]
            while len(entity_indices) < model.negative + 1:
                w = model.table[random.randint(model.table.shape[0])]
                if w != e2.index:
                    entity_indices.append(w)
            R = deepcopy(model.rel_mat[r.index])
            l2b = deepcopy(model.syn0[entity_indices]) # 2d matrix, k+1 x layer1_size
            fb = 1./ (1. + exp(-dot(l1,dot(R,l2b.T))))
            gb = (labels - fb) * alpha
            model.syn0[e1.index] += dot(dot(diag(gb),l2b),R.T).sum(axis = 0)
            model.syn0[entity_indices] += outer(g,dot(l1,R))
            model.rel_mat[r.index] += dot(tile(l1,(len(labels),1)).T,dot(diag(gb),l2b))
            return 1

        elif model.hs:
            Vwi = model.syn0[e1.index]
            neur0 = zeros(Vwi.shape)

            Vnw_ = deepcopy(model.syn1[e2.point])
            L = Vnw_.shape[0]
            R = model.rel_mat[r.index]
            fa = 1. / (1.+exp(-dot(Vwi,dot(R,Vnw_.T))))
            ga = (1 - e2.code - fa) * alpha
            model.syn1[e2.point] += outer(ga,dot(Vwi,R))
            model.rel_mat[r.index] += dot(tile(Vwi,(L,1)).T,dot(diag(ga),Vnw_))
            model.syn0[e1.index] += dot(dot(diag(ga),Vnw_),R.T).sum(axis = 0)
            return 1
        else:
            return 0

class Vocab(object):
    """A single vocabulary item, used internally for constructing binary trees (incl. both word leaves and inner nodes)."""
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"



class KB2Vec(utils.SaveLoad):
    def __init__(self,triples,size=100,alpha=0.025,min_count=0,min_count_rel = 0,sample=0,
                 seed=1,workers=1,min_alpha=0.0001,sg=1,hs=0,negative = 1, hashfxn=hash,iter=1):
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2entity_name = []
        self.index2rel_name = []
        self.sg = int(sg)
        self.layer1_size = int(size)
        if size%4 !=0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.seed = seed
        self.min_count = min_count
        self.min_count_rel = min_count_rel
        self.sample = 0
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.hashfxn = hashfxn
        self.iter = iter
        if triples is not None:
            self.build_vocab(triples)
            triples = utils.RepeatCorpusNTimes(triples,iter)
            self.train(triples)

    def make_table(self,table_size = 100000000,power = 0.75):
        """
        Create a table using stored vocabulary word counts for drawing random words in the negative
        sampling training routines.
        Called internally from `build_vocab()`.
        """
        logger.info("constructing a table with noise distribution from %i words" % len(self.vocab))
        # table (= list of words) of noise distribution for negative sampling
        vocab_size = len(self.index2entity_name)
        self.table = zeros(table_size, dtype=uint32)

        if not vocab_size:
            logger.warning("empty vocabulary in kb2vec, is this intended?")
            return

        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.vocab[word].count**power for word in self.vocab]))
        # go through the whole table and fill it up with the word indexes proportional to a word's count**power
        widx = 0
        # normalize count^0.75 by Z
        d1 = self.vocab[self.index2entity_name[widx]].count**power / train_words_pow
        for tidx in xrange(table_size):
            self.table[tidx] = widx
            if 1.0 * tidx / table_size > d1:
                widx += 1
                d1 += self.vocab[self.index2entity_name[widx]].count**power / train_words_pow
            if widx >= vocab_size:
                widx = vocab_size - 1


    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.
        """
        logger.info("constructing a huffman tree from %i entities" % len(self.vocab))

        # build the huffman tree for entities
        heap = list(itervalues(self.vocab))
        heapq.heapify(heap)
        for i in xrange(len(self.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(self.vocab), left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(self.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(self.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            logger.info("built huffman tree with maximum node depth %i" % max_depth)


    def build_vocab(self,triples):
        logger.info("collecting all words and their counts")
        vocab,vocab_rel = self._vocab_from(triples)
        #assign a unique index to each word
        self.vocab, self.vocab_rel, self.index2entity_name,self.index2rel_name = {},{},[],[]
        for entity,v in iteritems(vocab):
            if v.count >= self.min_count:
                v.index = len(self.vocab)
                self.index2entity_name.append(entity)
                self.vocab[entity] = v
        logger.info("total %i entity types after removing those with count<%s" % (len(self.vocab), self.min_count))

        for relation,v in iteritems(vocab_rel):
            if v.count >= self.min_count_rel:
                v.index = len(self.vocab_rel)
                self.index2rel_name.append(relation)
                self.vocab_rel[relation] = v
        logger.info("total %i relation types after removing those with count<%s" % (len(self.vocab_rel), self.min_count))

        if self.hs:
            # add info about enach word's Huffman encoding
            self.create_binary_tree()

        if self.negative:
            self.make_table()
        self.reset_weights()

    @staticmethod
    def _vocab_from(triples):
        triples_no,vocab = -1,{}
        vocab_rel = {}
        total_entities = 0
        total_relations = 0
        for triples_no, triple in enumerate(triples):
            if triples_no % 10000 == 0:
                logger.info("PROGRESS: at triple #%i, processed %i entities and %i entities types %i relations and %i relation types " %
                            (triples_no, total_entities, len(vocab), total_relations, len(vocab_rel)))

            e1, r, e2 = triple
            total_entities += 2
            total_relations += 1
            if e1 in vocab:
                vocab[e1].count += 1
            else:
                vocab[e1] = Vocab(count = 1)
            if e2 in vocab:
                vocab[e2].count += 1
            else:
                vocab[e2] =  Vocab(count = 1)
            if r in vocab_rel:
                vocab_rel[r].count += 1
            else:
                vocab_rel[r] = Vocab(count = 1)
        logger.info("collected %i entities types %i relations types from a kb of %i entities, %i relation and %i triples" %
                    (len(vocab), len(vocab_rel),total_entities,total_relations, triples_no + 1))
        return vocab, vocab_rel
    # subsampling and transform name to vocab
    def _prepare_triples(self,triples):
        for e1,r,e2 in triples:
            yield [self.vocab[e1],self.vocab_rel[r],self.vocab[e2]]


    def _get_job_triples(self,alpha,job,work,detR):
        if self.sg:
            return sum(train_triple_sg(self,triple,alpha,work,detR) for triple in job)

    def train(self,triples, total_triples=None, triples_count = 0, chunksize=1000):
        if not self.vocab or not self.vocab_rel:
            raise RuntimeError("you must first build entity and relation vocabulary before training the model")
        start,next_report = time.time(),[1.0]
        triples_count = [triples_count]
        total_triples = total_triples or int(sum(1 for v in triples))
        jobs = Queue(maxsize=2*self.workers)
        lock = threading.Lock()

        def worker_train():
            work = zeros(self.layer1_size, dtype=REAL)
            detR = zeros((self.layer1_size,self.layer1_size),dtype=REAL)
            # neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            while True:
                job = jobs.get()
                if job is None:
                    break
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * triples_count[0] / total_triples))
                job_triples = self._get_job_triples(alpha,job,work,detR)
                with lock:
                    triples_count[0] += job_triples
                    elapsed = time.time() - start
                    if elapsed>= next_report[0]:
                        logger.info("PROGRESS: at %.2f%% triplrs, alpha %.05f, %.0f triples/s" %
                            (100.0 * triples_count[0] / total_triples, alpha, triples_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(self._prepare_triples(triples), chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i triples took %.1fs, %.0f triples/s" %
            (triples_count[0], elapsed, triples_count[0] / elapsed if elapsed else 0.0))
        self.syn0norm = None
        return triples_count[0]

    def reset_weights(self):
        logger.info("resetting layer weights")
        self.syn0 = empty((len(self.vocab),self.layer1_size),dtype=REAL)
        self.rel_mat = empty((len(self.vocab_rel),self.layer1_size,self.layer1_size),dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.vocab)):
            random.seed(uint32(self.hashfxn(self.index2entity_name[i]+ str(self.seed))) )
            self.syn0[i] = ((random.rand(self.layer1_size) - 0.5) / self.layer1_size)

        for i in xrange(len(self.vocab_rel)):
            random.seed(uint32(self.hashfxn(self.index2rel_name[i]+ str(self.seed))) )
            self.rel_mat[i] = (random.rand(self.layer1_size, self.layer1_size) - 0.5) / (self.layer1_size**2)
        if self.hs:
            self.syn1 = zeros((len(self.vocab),self.layer1_size),dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.vocab),self.layer1_size),dtype=REAL)
        self.syn0norm = None

    def init_sims_entity(self, replace=False):
        """
        Precompute L2-normalized vectors.
        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!
        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.
        """
        if getattr(self, 'syn0norm', None) is None or replace:
            logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.syn0.shape[0]):
                    self.syn0[i, :] /= sqrt((self.syn0[i, :] ** 2).sum(-1))
                self.syn0norm = self.syn0
                if hasattr(self, 'syn1'):
                    del self.syn1

            else:
                self.syn0norm = (self.syn0 / sqrt((self.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)
            logger.info("finish normalization")
    def init_sims_relation(self,replace=False):
        if getattr(self, 'rel_mat_norm', None) is None or replace:
            logger.info("precomputing L2-norms of relation matrices")
            self.rel_mat_norm = empty(self.rel_mat.shape,dtype=REAL)

            for i in xrange(self.rel_mat.shape[0]):
                sum = sqrt((self.rel_mat[i] ** 2).sum(0).sum(0))
                self.rel_mat_norm[i] = (self.rel_mat[i]/sum).astype(REAL)

            logger.info("finish relation normalization")
    def most_similar_entity(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.
        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words, and corresponds to the `word-analogy` and
        `distance` scripts in the original word2vec implementation.
        Example::
          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]
        """
        self.init_sims_entity()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, string_types + (ndarray,))
                                else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, string_types + (ndarray,))
                                 else word for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                mean.append(weight * self.syn0norm[self.vocab[word].index])
                all_words.add(self.vocab[word].index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        dists = dot(self.syn0norm, mean)
        if not topn:
            return dists
        best = argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2entity_name[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def most_similar_relation(self,relation,topn = 10 ):
        self.init_sims_relation()

        in_index = self.vocab_rel[relation].index
        in_rel = self.rel_mat_norm[in_index]
        n = self.rel_mat.shape[0]
        dists = zeros(n,dtype=REAL)
        for i in range(n):
            dists[i] = ((self.rel_mat_norm[i] - in_rel)** 2).sum(0).sum(0)
        best = argsort(dists)[:topn+1]
        return [(self.index2rel_name[sim],float(dists[sim])) for sim in best if sim != in_index][:topn]

    def object_predict(self,subject,relations,topn = 10,chunksize = 1000):
        """
        Find the top-N most similar entities to (subj * rel-1 *rel-2 *...*rel-k)

        """
        self.init_sims_entity()

        if subject not in self.vocab:
            raise ValueError('entity %s not in vocabulary' % subject)

        start_time, next_report = time.time(),[1.0]
        if isinstance(relations,string_types):
            relations = [relations]
        m = ones(self.rel_mat[0].shape)
        for rel in relations:
            if rel in self.vocab_rel:
                m = dot(m,self.rel_mat[self.vocab_rel[rel].index])
        obj = dot(self.syn0[self.vocab[subject].index],m)
        obj = matutils.unitvec(obj).astype(REAL)
        jobs = Queue(maxsize=2*self.workers)
        results = Queue()
        entities_count = [0]
        lock = threading.Lock()

        def worker_compute_dists():
            while True:
                start = jobs.get()
                if start == None:
                    break
                end = min(start+chunksize,len(self.vocab))
                candidates = self.syn0norm[xrange(start,end)]
                dists = dot(candidates,obj)

                # topn = topn or end - start

                best = argsort(dists)[::-1][:topn+1]

                sub_index = self.vocab[subject].index
                result = [(self.index2entity_name[sim+start],float(dists[sim])) for sim in best if sim+start != sub_index]
                results.put(result)
                # elspsed = time.time() - start_time
                # with lock:
                #     entities_count[0] += end - start + 1
                #     if elspsed> next_report[0]:
                #         logger.info("PROGRESS: at %.2f%% entities,%.0f entities/s" %
                #                     (100* entities_count[0]/len(self.vocab), entities_count[0]/ elspsed if elspsed else 0.0))
                #         next_report[0] = elspsed+1.0




        workers = [threading.Thread(target=worker_compute_dists,) for _ in range(self.workers)]

        for thread in workers:
            thread.daemon = True
            thread.start()


        group_num = len(self.vocab)/chunksize + (1 if len(self.vocab)%chunksize!=0 else 0)
        for start in xrange(group_num):
            jobs.put(start*chunksize)
        for _ in xrange(self.workers):
            jobs.put(None)
        for thread in workers:
            thread.join()
        candidates = []
        while not results.empty():

            result = results.get_nowait()
            candidates.extend(result)
        return sorted(candidates,key=lambda item:item[1],reverse=True)[:topn]



class KBTriple(object):

    # only remain relation name (without url prefix and angle brackets)
    tail = lambda self,url : utils.to_unicode(os.path.basename(url)[:-1])

    def __init__(self,source):
        self.source = source
        import re
        self.p = re.compile(r"(?P<e1><[^>]+>)[ ]+(?P<r><[^>]+>)[ ]+(?P<e2>((<[^>]+>)|(\"[^\"]+\")))")

    def parse(self,line):
        """
        :parameter
        line: string
            one line triple
        :return
            a list which contains e1,r,e2 without url prefix
        """
        m = self.p.match(line)
        if m != None:
            e1,r,e2 = m.group('e1'),m.group('r'),m.group('e2')
            if e2.startswith('\"'):
                e2 = r
            return [self.tail(e1),self.tail(r),self.tail(e2)]

    def __iter__(self):

        # try:
        #     self.source.seek(0)
        #     for line in self.source:
        #         yield utils.to_unicode(line).split()
        #
        # except AttributeError:
            # If it didn't work like a file, use it as a string filename
        with utils.smart_open(self.source) as fin:
            for line in fin:
                t3 = self.parse(line)
                if t3!=None:
                    yield t3


# python kb2vec.py kb_path output_model_path
def func1():
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))
    logging.info("using optimization %s" % FAST_VERSION)

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    infile = sys.argv[1]
    # from gensim.models.word2vec import Word2Vec  # avoid referencing __main__ in pickle

    seterr(all='raise')  # don't ignore numpy errors

    # model = Word2Vec(LineSentence(infile), size=200, min_count=5, workers=4)
    model = KB2Vec(KBTriple(infile), negative= 10, hs = 0, size=10, workers=2,iter = 10)

    if len(sys.argv) > 2:
        outfile = sys.argv[2]
        model.save(outfile + '.model')
        # model.save_word2vec_format(outfile + '.model.bin', binary=True)
        # model.save_word2vec_format(outfile + '.model.txt', binary=False)



    logging.info("finished running %s" % program)

def func2():
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))
    logging.info("using optimization %s" % FAST_VERSION)

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    infile = sys.argv[1]

    seterr(all='raise')
    model = KB2Vec.load(infile)
    print model.object_predict('Apple_Inc.',['keyPerson'],topn=10,chunksize=100)



# Example: ./word2vec.py ~/workspace/word2vec/text8 ~/workspace/word2vec/questions-words.txt ./text8
if __name__ == "__main__":
    t1 = time.time()
    func1()
    print time.time() - t1



