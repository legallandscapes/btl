# -*- coding: UTF-8 -*-
"""
EXAMPLE.py
``````````
Shows how to use the BTL library.
"""
import corpus
import topicmodel
import btl

import numpy
import scipy

#print("[!] Building corpus...");

# Create a tokenizer (you can write your own as well)
tokenizer = corpus.Tokenizer(
        stop_tokens = ["l","stat","pub"],

        use_lexical_smoothing = True,        
        use_stemming          = True,
        use_pos_tagging       = True
);

db_path = "/home/legal_landscapes/public_html/beta/usc.db";

(dictionary, dtm) = corpus.from_database(tokenizer.tokenize, db_path, """
        SELECT text
        FROM nodes
""");

dictionary.save("usc.dictionary");
numpy.savez("usc.dtm", dtm);

#print("[!] Running LDA model...");
(theta, phi) = topicmodel.lda(dtm, num_topics=100, num_passes=10);

numpy.savez("usc.theta", theta);
numpy.savez("usc.phi", phi);

#print("[!] Computing similarity matrix...");
SIM = btl.similarity_matrix(theta, M_T=10, M_O=10);

numpy.savez("usc.sim", SIM);

#print("[!] Computing citation matrix...");
CITE = btl.citation_matrix(db_path, """
        SELECT n0.rowid, n1.rowid 
        FROM edges
        JOIN nodes as n0
                ON source_url = n0.url
        JOIN nodes as n1
                ON target_url = n1.url
""");

numpy.savez("usc.cite", CITE);

print("[!] Computing normalized weighted transition matrix...");
TRAN = btl.weighted_transition_matrix(SIM, CITE, 0.33, 0.33, 0.33);

print("[!] Saving matrices...");
dictionary.save("usc.dictionary");

numpy.savez("usc.dtm", dtm);

numpy.savez("usc.theta", theta);
numpy.savez("usc.phi", phi);

numpy.savez("usc.sim", SIM);
numpy.savez("usc.cite", CITE);
numpy.savez("usc.tran", TRAN);
