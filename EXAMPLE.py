# -*- coding: UTF-8 -*-
"""
EXAMPLE.py
``````````
This file provides an example of how to use the BTL Python library.
In this example, an external SQLite3 database is being queried
to provide the necessary documents to form the corpus.

Change this as necessary for your own database, or form the corpus
from a directory of files or a list of strings using the various
helper functions in corpus.py.
"""
import corpus
import topicmodel
import btl

import numpy
import scipy

"""
Helper functions
"""
def save_sparse(filename, sparse):
        numpy.savez(filename, 
                data=sparse.data, 
                indices=sparse.indices, 
                indptr=sparse.indptr,
                shape=sparse.shape
        );

def load_sparse(filename):
        loader = numpy.load(filename);

        return scipy.sparse.csc_matrix(
                (loader["data"], loader["indices"], loader["indptr"]), 
                shape=loader["shape"]
        );


"""
The example will process an entire corpus all in one run.
You can stop the process at any point or load data from
a file and begin at any point.
"""

# Create a new tokenizer
tokenizer = corpus.Tokenizer(
        stop_tokens = ["l","stat","pub"],

        use_lexical_smoothing = True,        
        use_stemming          = True,
        use_pos_tagging       = True
);

# Create a dictionary and DTM from a database 
db_path = "/home/legal_landscapes/public_html/beta/usc.db";

(dictionary, DTM) = corpus.from_database(tokenizer.tokenize, db_path, """
        SELECT text
        FROM nodes
""");

# Create the THETA and PHI matrices from the LDA topic modeler
(THETA, PHI) = topicmodel.lda(dtm, num_topics=100, num_passes=10);

# Create the similarity matrix
SIM = btl.similarity_matrix(THETA, M_T=10, M_O=10);

# Create the citation matrix
CITE = btl.citation_matrix(db_path, """
        SELECT n0.rowid, n1.rowid 
        FROM edges
        JOIN nodes as n0
                ON source_url = n0.url
        JOIN nodes as n1
                ON target_url = n1.url
""");

# Create the transition matrix 
TRAN = btl.weighted_transition_matrix(SIM, CITE, 0.33, 0.33, 0.33);

# Create the rank matrix 
RANK = btl.rank_matrix(TRAN, 1/3.0);

# Create the final distance matrix 
DIST = btl.distance_matrix(RANK, 2);

# Save all the model data 
dictionary.save("usc.dictionary");
save_sparse("usc.sparse.dtm", DTM);
numpy.save("usc.theta", THETA);
numpy.save("usc.phi", PHI);
save_sparse("usc.sparse.sim", SIM);
save_sparse("usc.sparse.cite", CITE);
save_sparse("usc.sparse.tran", TRAN);
save_sparse("usc.sparse.rank", RANK);
save_sparse("usc.sparse.dist", DIST);


