# -*- coding: UTF-8 -*-
"""
topic.py
````````
Compute similarity with a Latent Dirichlet Allocation (LDA) topic model, 
which determines the distribution of TOPICS (collections of terms from
the dictionary) in each document.

"""
import sys

# The 'lda' library
sys.path.append("./lib")

import lda as liblda # avoid collision with 'lda' function 

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
# Topic modeling 
###############################################################################

def lda(dtm, **kwargs):
        """
        Run an LDA topic model using the provided document-term matrix

        Arguments:
                @dtm   : Document-topic matrix
                @kwargs: Additional options
        Return:
                (theta, phi), where theta is the document-topic distribution,
                and phi the topic-word distribution.
        Options:
                num_topics (int): How many topics to model 
                num_passes (int): How many times to iterate the model
        """
        num_topics = kwargs.pop("num_topics", 100);
        num_passes = kwargs.pop("num_passes", 10);

        LOG.info("Building LDA model num_topics="+str(num_topics)+", num_passes="+str(num_passes));

        # Instantiate the LDA model using the 'lda' library.
        model = liblda.LDA(
                n_topics = num_topics,
                n_iter   = num_passes
        );

        LOG.info("Fitting LDA model to "+str(dtm.shape[0])+"x"+str(dtm.shape[1])+" DTM");

        # Fit the model to the document-term matrix (takes a while).
        model.fit(dtm);

        LOG.info("Done");

        # Returns (theta, phi), to use the terminology in the literature.
        return (model.doc_topic_, model.topic_word_);


def format_theta(theta_topic_row, limit=10):
        """
        Convert full theta row (topic) into compact form

        The result will be sorted by probability (high to low).
        """
        topics = theta_topic_row;
        topics = [(i, val) for i, val in enumerate(topics)];
        topics = sorted(topics, key=lambda x: x[1], reverse=True);
        topics = topics[:limit];

        return topics;
