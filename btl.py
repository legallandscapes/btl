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
# Compute the similarity matrix
###############################################################################
def similarity_matrix(theta=None, M_T=10, M_O=10):

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

                        LOG.info("1. Arrays [%d/%d][%d/%d]", doc_id, len(theta), i, len(topics));

        #
        # Build this matrix thing to join docs to "similar" docs
        #
        MATRIX = {};

        for doc_id in DOC_TOPICS:

                MATRIX[doc_id] = [];

                for i in range(len(DOC_TOPICS[doc_id])):
                        topic_id = DOC_TOPICS[doc_id][i];

                        MATRIX[doc_id].append(TOPIC_DOCS[topic_id].values());

                        LOG.info("MATRIX [%d/%d][%d/%d]", doc_id, len(theta), i, len(DOC_TOPICS[doc_id]));


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

                                LOG.info("3. W [%d][%d]", doc_id_A, doc_id_B); 
                
        #
        # Build the similarity matrix
        #
        T_sim = scipy.sparse.dok_matrix((len(theta), len(theta)), dtype=numpy.float);

        for a in W:
                total = 0;
                for b in W[a]:
                        if W[a][b] > 0:
                                total += W[a][b];
                        
                        LOG.info("4a. T_sim [%d][%d]", a, b); 
                        
                for b in W[a]:
                        if W[a][b] > 0:
                                T_sim[a,b] = float(W[a][b])/total;
                
                        LOG.info("4b. T_sim [%d][%d]", a, b); 

        return T_sim.tocsc();

###############################################################################
# Compute the citation matrix  
###############################################################################
def citation_matrix(db_path, db_query):
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
        LOG.info("Building normalized weighted transition matrix");

        # The weighted sum
        T = (w_sim*M_sim) + (w_cite*M_cite) + (w_cited_by*M_cite.transpose());

        # Fill the diagonal with 0. 
        # Newer versions of numpy can use T = numpy.fill_diagonal(T, 0);
        for i in range(T.shape[0]):
                T[i,i] = 0;

        # Get a list of the sums of values in each row.
        row_sums = T.sum(axis=1);

        # If a row sums to 0, replace this sum with 1, to prevent 
        # division by 0 in the next step. The result will still be 
        # 0 (the correct value) in the normalization.
        row_sums[row_sums == 0] = 1;

        # Uses numpy broadcasting to act elementwise on @row_sums
        T = T / row_sums; 

        return T;


###############################################################################
# Compute the distance matrix
###############################################################################

#def distance_matrix(T, r):
 




