# -*- coding: UTF-8 -*-
"""
btl.py
``````
"""

import sys

import os
import sqlite3  # SQLite is part of the standard library 
import numpy    # numpy arrays used to store DTM matrix 
import btl_corpus
from btl_similarity import BTL_corpus, BTL_tokenizer, btl_similarity_matrix
import btl_similarity
import logging

logging.basicConfig(format="%(asctime)-15s %(message)s")
LOG = logging.getLogger("BTL")


def from_list(tokenizer, text_list):

        docs = [];
        i    = 0;

        for t in text_list:
                docs.append(tokenizer.tokenize(t));

                i += 1;
                if not i % 100:
                        print("adding %d" % (i));

        return BTL_corpus(documents=docs);

def from_directory(tokenizer, directory_path):

        docs = [];
        i    = 0;

        for filename in os.listdir(directory_path): 
                text = open(filename).read();
                docs.append(tokenizer.tokenize(text));

                i += 1;
                if not i % 100:
                        print("adding %d" % (i));

        return BTL_corpus(documents=docs);

def from_database(tokenizer, path=None, query=None):

        docs = [];
        i    = 0;

        # Open a connection to the provided database file
        try: 
                conn = sqlite3.connect(path);
        except sqlite3.Error as e:
                print "SQLITE3 error: ", e.args[0];

        conn.text_factory = bytes; # Process non-ASCII characters

        db = conn.cursor(); 
        db.execute(query);

        for row in db:
                docs.append(tokenizer.tokenize(row[0]));

                i += 1;
                if not i % 100:
                        print("adding %d" % (i));

        return BTL_corpus(documents=docs);



#dic = btl_similarity.unpickle_it("usctest.dic");

#import pprint

#pprint.pprint(dic);

#exit();


tokenizer = BTL_tokenizer(
        stop_tokens = ["l","stat","pub"],

        use_lexical_smoothing = True,        
        use_stemming          = True,
        use_pos_tagging       = False,
);


corpus = from_database(tokenizer, 
        path="/home/legal_landscapes/public_html/beta/usc.db", 
        query="SELECT text FROM nodes WHERE text != '(Text Unavailable)' LIMIT 10000"
);


print("[!] Saving corpus...");
corpus.save(dictionary_path="usctest.dic", dtm_path="usctest.dtm");

print("[!] Running LDA model...");
model = corpus.lda(num_topics=100, num_passes=10);

print("[!] Saving LDA model theta and phi...");
model.save(phi_path="usctest.phi", theta_path="usctest.theta");

import pprint

print("[!] Computing similarity matrix...");
T_sim = btl_similarity_matrix(model.theta(), M_T=10, M_O=10);

#T_sim = model.get_similarity_matrix(10, 10);
#pprint.pprint(T_sim);

print("[!] Saving similarity matrix...");
numpy.savez("tsim.numpytxt", T_sim);



