# -*- coding: UTF-8 -*-
"""
btl.py
``````
Build all the big matrices
"""
import sys

# SortedDict is used for the list of high probability docs (Fix this)
sys.path.append("./lib");

from sortedcontainers import SortedDict

import sqlite3
import numpy    # numpy arrays used to store DTM matrix 
import scipy    # sparse arrays etc

import topicmodel

###############################################################################
# Logging  
###############################################################################
import logging

# create logger
LOG = logging.getLogger(__name__);
LOG.setLevel(logging.INFO);

# create console handler 
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create file handler 
fh = logging.FileHandler("btl.log")
fh.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to handlers 
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add handlers to logger
LOG.addHandler(ch)
LOG.addHandler(fh)

###############################################################################
# Save and load CSR matrices 
###############################################################################
def save_csr(path, matrix):
        numpy.savez(path, 
                data    = matrix.data, 
                indices = matrix.indices,
                indptr  = matrix.indptr,
                shape   = matrix.shape
        );

def load_csr(path):
        archive = numpy.load(path);

        return scipy.sparse.csr_matrix(
                (archive["data"], archive["indices"], archive["indptr"]), 
                shape = archive["shape"]
        );


###############################################################################
# Compute the similarity matrix
###############################################################################
def similarity_matrix(theta=None, M_T=10, M_O=10):
        """
        Compute the similarity matrix for a corpus, given 
        the distribution of topics therein.

        Arguments:
                @theta: The document-topic distribution from LDA
                @M_T  : The number of topics to consider
                @M_O  : The number of other documents (opinions) to consider 
        Return:
                NxN sparse matrix in CSR format (see function body for
                explanation).
        Notes:
                This needs to be cleaned up bigtime, it's a waste.
        """

        LOG.info("Building similarity matrix M_T="+str(M_T)+" M_O="+str(M_O));

        # doc_id => M_T highest-probability topic_ids 
        DOC_TOPICS = {};

        # topic_id => M_O highest-probability doc_ids 
        TOPIC_DOCS = {};

        # For each document in the corpus 
        for doc_id in range(len(theta)): 

                topics = topicmodel.format_theta(theta[doc_id]);

                DOC_TOPICS[doc_id] = [];

                for i in range(len(topics)):

                        top_id = topics[i][0];
                        top_pr = topics[i][1];

                        #
                        # Build the collection of highest-probability
                        # documents for each topic.
                        #
                        if top_id not in TOPIC_DOCS:
                                TOPIC_DOCS[top_id] = SortedDict();
                                TOPIC_DOCS[top_id][top_pr] = doc_id;

                        # Don't bother attempting to insert if the probability
                        # is less than the lowest already in the collection.
                        elif top_pr >= TOPIC_DOCS[top_id].peekitem(0)[0]:

                                if top_pr not in TOPIC_DOCS[top_id]:
                                        TOPIC_DOCS[top_id][top_pr] = doc_id;
                                else:
                                        # If two docs have an equal probability 
                                        # of expressing the topic, then which 
                                        # should we favor? We can only choose a 
                                        # certain number. Ignore for now.
                                        LOG.warn("Equal probabilities.");

                                if len(TOPIC_DOCS[top_id]) > M_O:
                                        # Remember, dict is sorted, so this 
                                        # will only discard the least one.
                                        TOPIC_DOCS[top_id].popitem(last=False);
                                        
                        #
                        # Build the collection of highest-probability 
                        # topics for each document.
                        #
                        if i < M_O:
                                DOC_TOPICS[doc_id].append(top_id);

        #
        # Build this matrix thing to join docs to "similar" docs
        #
        MATRIX = {};

        for doc_id in DOC_TOPICS:

                MATRIX[doc_id] = [];

                for i in range(len(DOC_TOPICS[doc_id])):
                        topic_id = DOC_TOPICS[doc_id][i];

                        MATRIX[doc_id].append(TOPIC_DOCS[topic_id].values());

        #
        # Build dictionary to count occurrences. 
        #
        W = {};

        for doc_id_A in DOC_TOPICS:
                W[doc_id_A] = {};

                # Count occurrences of doc_id_B in matrix of doc_id_A 
                for i in range(len(MATRIX[doc_id_A])):
                        for j in range(len(MATRIX[doc_id_A][i])):

                                doc_id_B = MATRIX[doc_id_A][i][j];

                                if doc_id_B not in W[doc_id_A]:
                                        W[doc_id_A][doc_id_B] = 1;
                                else:
                                        W[doc_id_A][doc_id_B] += 1;
        #
        # Build the similarity matrix
        #
        # FIXME: Why dok?
        T_sim = scipy.sparse.dok_matrix((len(theta), len(theta)), dtype=numpy.float);

        for a in W:
                total = 0;
                for b in W[a]:
                        if W[a][b] > 0:
                                total += W[a][b];
                        
                for b in W[a]:
                        if W[a][b] > 0:
                                T_sim[a,b] = float(W[a][b])/total;
                
        return T_sim.tocsr();

###############################################################################
# Compute the citation matrix  
###############################################################################
def citation_matrix(db_path, db_query):
        """
        Compute the citation matrix for a corpus, given a
        database query.

        Arguments:
                @db_path : Path to SQLite3 database file
                @db_query: Query to run
        Return:
                NxN sparse matrix in CSR format, with M[i][j] = 1
                if and only if document i cites document j.
        Notes:
                The argument @db_query when applied to the database
                at @db_path should return a 2-column result set,
                where ids in column 0 cite ids in column 1.
        """
        LOG.info("Building citation matrix");

        # Open a connection to the provided database file
        try: 
                conn = sqlite3.connect(db_path);
        except sqlite3.Error as e:
                LOG.error("SQLITE3 error: ", e.args[0]);

        conn.text_factory = bytes; # Process non-ASCII characters

        db = conn.cursor(); 

        count = 0;

        # TODO: write up format for database or else factor this out.
        db.execute("SELECT count(*) from nodes");

        for row in db:
                count = row[0];

        # Create the (sparse) citation matrix 
        M_cite = scipy.sparse.lil_matrix((count, count), dtype=numpy.int8);

        db.execute(db_query);

        for row in db:
                M_cite[int(row[0]),int(row[1])] = 1;

        return M_cite.tocsr();

###############################################################################
# Compute the weighted transition matrix 
###############################################################################
def weighted_transition_matrix(M_sim, M_cite, w_sim, w_cite, w_cited_by):
        """
        Compute the weighted transition matrix given the three
        weights for similarity, citation, and -citation.

        Arguments:
                @M_sim     : Similarity matrix
                @M_cite    : Citation matrix
                @w_sim     : (float) Similarity edge weight from 0-1
                @w_cite    : (float) Citation edge weight from 0-1
                @w_cited_by: (float) Cited-by edge weight from 0-1
        Return:
                Transition matrix in CSR format.

        Notes:
                M_sim, M_cite, and the returned matrix should all be
                NxN, where N is the number of documents in the corpus.

                @w_sim, @w_cite, and @w_cited_by should add up to 1. 
        """
        LOG.info("Building normalized weighted transition matrix");

        # The weighted sum
        T = (w_sim*M_sim) + (w_cite*M_cite) + (w_cited_by*M_cite.transpose());

        # Faster for modification
        T = T.tolil();

        # Fill the diagonal with 0. 
        # Newer versions of numpy can use T = numpy.fill_diagonal(T, 0);
        for i in range(T.shape[0]):
                T[i,i] = 0;

        # Get a list of the sums of values in each row.
        row_sums = T.sum(axis=1);

        # If a row sums to 0, replace this sum with 1, to prevent 
        # division by 0 in the next step. The result will still be 
        # 0 (the correct value) in the normalization.
        row_sums[row_sums == 1] = 1;

        # Uses numpy broadcasting to act elementwise on @row_sums
        T = T / row_sums; 

        # Back to CSR format
        return scipy.sparse.csr_matrix(T);

###############################################################################
# Compute the rank matrix
###############################################################################
def rank_matrix(T, r):
        """
        Compute a rank matrix from a non-negative square matrix, sp.
        a matrix of transition probabilities for a Markov process. 

        Arguments:
                @T: Transition matrix
                @r: Stopping probability
        Return:
                Rank matrix
        Notes:
                Formally, the rank matrix R is defined
                
                        R = \sum_{k=0}^{\infty} r^k * T^k 

                so that the value R(a,b) is the expected number of
                steps for a random walk beginning at a to end at b.

                This infinite sum is in fact the power series expansion
                of the matrix

                        Y(r,T) := (I - rT)^{-1},

                where I is the identity matrix matching T.

                This matrix is called the resolvent of T, and has other
                applications as well, as well as technical constraints
                which are satisfied in this use case, so let's not go
                into them, hmm?
        """
        n = T.shape[0] if hasattr(T, "shape") else len(T);

        # Get the identity matrix
        #I = numpy.eye(n);
        I = scipy.sparse.identity(n, format="csc");
        print("[!] Done building CSC identity matrix");

        # Obtain the resolvent
        # Warning: This may be very unkind to other users on the system...
        #R = numpy.linalg.solve(I - r*T, I);

        R = scipy.sparse.linalg.spsolve(I - r*T, I);
        print("[!] Done solving for R");

        #return scipy.sparse.csc_matrix(R);
        return R;

###############################################################################
# Compute the PageDist matrix 
###############################################################################
def distance_matrix(R, p=2): 
        """
        Compute the PageDist matrix, given the rank matrix (resolvent)
        of the normalized weighted transition probability matrix.

        Arguments:
                @R: Rank matrix / resolvent of T
                @p: p-norm (default=2, Euclidean norm)
        Return:
                PageDist matrix of size equal to R's
        Notes:
                We define
                        PageDist(a,b) := ||R(a, ) - R(b, )||_p
                                       
                which is equal to
                        (\sum_{x\in R} [R(a,x) - R(b,x)]^p)^{1/p},

                that is, the p-norm.
        """

        # Number of rows/cols in the matrix R.
        n = R.shape[0] if hasattr(R, "shape") else len(R);

        # 1/p
        p_inv = 1/1.0*p;

        # Will be the result
        D = None;

        for i in range(n):
                # Get the i-th row of R as a 1xn matrix
                row = R[i,:]; 

                # Make a new matrix where every row is equal to @row
                tile_i = numpy.tile(row, [n,1]);

                # Compute the p-norm for R(x,y) - R(i,z), for x,y,z in n. 
                dist_i = numpy.sum(abs(R - tile_i)**p, axis=1) ** p_inv;

                if D is None:
                        D = dist_i;
                else: 
                        D = numpy.concatenate([D, dist_i]);

        return D;

