# -*- coding: UTF-8 -*-
"""
similarity.py
`````````````
Form a corpus object from a collection of documents presented as

(1) SQLite database, 
(2) directory of flat files, or 
(3) list of strings.

The documents are not themselves stored, but are rather processed
into two data structures:

DICTIONARY --
    The set of words occurring in the documents is called the vocabulary. 
    In practice, it is a good idea to fix a reasonable maximum for the
    vocabulary size, and discard the least frequently seen words to meet
    the size requirement.

    Each word in the vocabulary is assigned an integer-valued ID to represent 
    it, and a DICTIONARY is used to map between a word and its ID. 

DOCUMENT-TERM MATRIX (DTM) --
    Contains one row for each document in the collection, and one
    column for each entry in the dictionary. 

    The value of DTM[i][j] is equal to the number of times the j-th
    entry in the DICTIONARY has occurred in the i-th document of the 
    collection.

The DTM is the input to a Latent Dirichlet Allocation (LDA) topic model, 
which determines the distribution of TOPICS (collections of terms from
the dictionary) in each document.

"""

import sys

# NLTK used for pre-processing of the text corpus
sys.path.append("lib/nltk")

# NLTK used for pre-processing of the text corpus
sys.path.append("lib/lda")

# SortedDict is used for the list of high probability docs (Fix this)
sys.path.append("lib/sorted_containers");

from sortedcontainers import SortedDict

import os       # Used to access files
import re       # Regular expressions used in tokenizer
import pickle   # Used for saving and loading
import sqlite3  # SQLite is part of the standard library 
import nltk     # Only used for part-of-speech tagging 
import numpy    # numpy arrays used to store DTM matrix 
import lda      # runs the LDA topic model
import scipy

import logging

logging.basicConfig(format="%(asctime)-15s %(message)s")
LOG = logging.getLogger("BTL")

# Ensure that the POS tagging database is available.
# nltk.download("averaged_perceptron_tagger");

# Tell NLTK where it can find the data for Part-of-speech tagging. 
nltk.data.path.append("/home/legal_landscapes/btl/lda/data/nltk_data/");


# English stopword list from Stone, Denis, Kwantes (2010)
STOPWORDS = """
a about above across after afterwards again against all almost alone along already also although always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at back be
became because become becomes becoming been before beforehand behind being below beside besides between beyond bill both bottom but by call can
cannot cant co computer con could couldnt cry de describe
detail did didn do does doesn doing don done down due during
each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except few fifteen
fify fill find fire first five for former formerly forty found four from front full further get give go
had has hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how however hundred i ie
if in inc indeed interest into is it its itself keep last latter latterly least less ltd
just
kg km
made make many may me meanwhile might mill mine more moreover most mostly move much must my myself name namely
neither never nevertheless next nine no nobody none noone nor not nothing now nowhere of off
often on once one only onto or other others otherwise our ours ourselves out over own part per
perhaps please put rather re
quite
rather really regarding
same say see seem seemed seeming seems serious several she should show side since sincere six sixty so some somehow someone something sometime sometimes somewhere still such system take ten
than that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin third this those though three through throughout thru thus to together too top toward towards twelve twenty two un under
until up unless upon us used using
various very very via
was we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would yet you
your yours yourself yourselves
"""
STOPWORDS = [w for w in STOPWORDS.split() if w];


def pickle_it(name, obj):
        with open(name, "wb") as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def unpickle_it(name):
        with open(name, "rb") as f:
                return pickle.load(f)


class BTL_tokenizer:
        """ 
        Pre-processing a text 
        =====================
        
        Prior to the texts being processed by the LDA model, they
        are pre-processed in order to smooth out non-semantic 
        features and improve similarity performance.
        
        These are, in order:
         
        1. Lexical smoothing 
        --------------------
              a. All sequences of whitespace become a single space
              b. All non-ASCII and punctuation characters are removed
              c. All letters are made lowercase
       
        2. Part-of-speech (POS) filtering 
        ---------------------------------
              a. Words that are not nouns are removed
        
        3. Stop word removal 
        --------------------
              a. Tokens in NLTK English stop word list are removed. 
              b. Tokens in user-provided stop word list are removed.
               
        4. Stemming
        -----------
              a. Related words replaced with a common stem 
        """ 

        # Match all sequences of whitespace, "--", or pre-escaped \n and \t.
        match_whitespace = re.compile("(\s|--|\\\\n|\\\\t)+");

        # Disallow characters matching this regex (note [^...])
        match_disallowed = re.compile("[^a-zA-Z0-9-_@#%&=\$\*\+\|\s]+");

        # List of English stop words. May combine with user's (see __init__())
        stop_tokens = STOPWORDS;

        # Instantiate an object to run Porter's stemming algorithm
        stemmer = nltk.stem.porter.PorterStemmer();

        """ 
        smooth_lexical()
        ---------------- 
        @string: Input string to be manipulated
        Return : The smoothed string

        EXAMPLE:
                "are  you\\nwell--sailor John?" => "are you well sailor john"
        """
        def smooth_lexical(self, string):

                # Replace all whitespace sequences with single space
                string = self.match_whitespace.sub(" ", string);

                # Remove all undesirable characters from string
                string = self.match_disallowed.sub("", string);

                # Convert all letters to lowercase
                string = string.lower();

                return string;

        """ 
        smooth_part_of_speech()
        ----------------------- 
        @tokens: Input as an array of words 
        Return : Some subset of the tokenized_string 

        EXAMPLE:
                ["are","you","well","sailor","john"] => ["you","sailor","john"]
        """
        def smooth_part_of_speech(self, tokens):

                tagged = nltk.pos_tag(tokens);

                return [w[0] for w in tagged if w[1][0] == "N"];

        """ 
        smooth_stop_tokens()
        -------------------- 
        @tokens: Input as an array of tokens 
        Return : Some subset of the tokenized_string 

        EXAMPLE:
                ["you","sailor","john"] => ["sailor","john"]
        """
        def smooth_stop_tokens(self, tokens):

                return [w for w in tokens if not w in self.stop_tokens];

        """ 
        smooth_stem_tokens()
        -------------------- 
        @tokens: Input as an array of words 
        Return : The input array, with each word stemmed 

        EXAMPLE:
                ["sailor","john"] => ["sail","john"]
        """
        def smooth_stem_tokens(self, tokens): 

                return [self.stemmer.stem(w) for w in tokens];

        """
        tokenize()
        ----------
        Convert a string into a smoothed token list

        @string: Input string
        Return : Processed token list (array of strings)
        """
        def tokenize(self, string):

                if self.do_lex:
                        string = self.smooth_lexical(string);

                tokens = string.split();

                if self.do_pos:
                        tokens = self.smooth_part_of_speech(tokens);

                if self.do_stop:
                        tokens = self.smooth_stop_tokens(tokens);

                if self.do_stem:
                        tokens = self.smooth_stem_tokens(tokens);

                return tokens;

        """
        __init__()
        ----------
        Create a new BTL_tokenizer object

        @kwargs: Associative array of options
        Return : self

        Options:
                (BOOL) use_lexical_smoothing 
                (BOOL) use_pos_tagging
                (BOOL) use_stemming 
                (BOOL)|(ARRAY OF STRING) stop_tokens 
        """
        def __init__(self, **kwargs):
                toks = kwargs.pop("stop_tokens", True);

                self.do_stop = (toks is not False);

                if isinstance(toks, str) and os.path.exists(toks): 
                        self.stop_tokens += [l.rstrip("\n") for l in open(toks)];
                elif isinstance(toks, list):
                        self.stop_tokens += toks;

                self.do_lex  = kwargs.pop("use_lexical_smoothing", True);
                self.do_pos  = kwargs.pop("use_pos_tagging", True);
                self.do_stem = kwargs.pop("use_stemming", True);


class BTL_dictionary:
        vocabulary = {}; # token => token_id (note this mapping is 1:1)
        frequency  = {}; # token_id => number of docs token has occurred in

	def __init__(self, vocabulary=None):
		if vocabulary is not None:
			self.vocabulary = vocabulary;

        def __len__(self):
                return len(self.vocabulary);

        def __iter__(self):
                return iter(self.keys());

        def __getitem__(self, word):
                return self.vocabulary[word]; 

        def keys(self):
                return list(self.vocabulary.values());

        def save(self, filename):
                """Save the dictionary as 2-column text

                Args:
                        @self    : The dictionary object (automatic) 
                        @filename: File to save the dictionary to

                Return:
                        Nothing
                """
                f = open(filename, "w");

                for tok, tok_id in self.vocabulary.items():
                        f.write(str(tok) + " " + str(tok_id) + "\n")

                f.close()

        def load(self, filename):
                """Load the dictionary from a 2-column text file

                Args:
                        @self    : The dictionary object (automatic) 
                        @filename: File to load the dictionary from 

                Return:
                        Nothing
                """
                f = open(filename, "w");

                for line in f:
                        (tok, tok_id) = line.split();

                        if not isinstance(tok, unicode):
                                tok = unicode(tok, "utf-8");
                        
                        self.vocabulary[tok] = tok_id; 

                f.close();

        def lookup(self, tok_id): 
                """Lookup a token from a token id.

                Args:
                        @self  : The dictionary object (automatic) 
                        @tok_id: ID of the token to look up

                Return:
                        A unicode string (the token), or None. 

                NOTE:
                This reverse lookup is reasonably quick and efficient
                due to dict.values() and dict.keys() returning lists
                that only reference dict rather than copying the data. 

                Still, if this is used often enough, a reverse lookup
                table should be added. 
                """
                # Get the location of the value @tok_id in the vocabulary.
                tok_index = self.vocabulary.values().index(tok_id);

                # Return the key corresponding to that same location.
                return self.vocabulary.keys()[tok_index];

        def grow(self, token_list):
                """Add tokens to the dictionary from a token list 
                        
                Args:
                        @self      : The dictionary object (automatic) 
                        @token_list: List of unicode strings

                Return:
                        Nothing

                NOTE:
                By convention, the @token_list provided to each call of 
                .grow() is considered to represent an individual document. 
                
                For the purposes of counting token document frequencies, 
                the dictionary will treat distinct calls to .grow() as
                distinct documents, even if the values of @token_list are
                identical, and no new tokens are added to the dictionary.
                
                Likewise, if a single document is provided to the dictionary
                across multiple calls to .grow(), it will not be treated
                as a single document. 

                Outside of the calculation of token document frequencies,
                there will be no adverse effect.
                """
                # Fast way to remove duplicate tokens.
                token_list = {}.fromkeys(token_list).keys();

                for tok in token_list:
                        # Prevent KeyErrors by ensuring tokens are unicode 
                        if not isinstance(tok, unicode):
                                tok = unicode(tok, "utf-8");

                        if tok not in self.vocabulary:
                                tok_id = len(self);

                                self.vocabulary[tok] = tok_id;
                                self.frequency[tok_id] = 1;
                        else:
                                self.frequency[token_id] += 1;

        def bow(self, token_list):
                """Convert a token list into a bag of words 

                Args:
                        @self      : The dictionary object (automatic) 
                        @token_list: List of Unicode strings.
                        
                Return: 
                        Numpy array of token frequencies. 

                NOTE 1:
                The returned array is of the form

                        [f_0, f_1, f_2, ..., f_n],

                where n = len(self) is the number of tokens in the 
                dictionary, and f_i is the number of times that the
                token having id 'i' (in the dictionary) occurs in 
                @token_list.

                NOTE 2:
                The length of the returned array is equal to the
                length of the dictionary, even though in practice
                most values will be 0. 

                NOTE 3:
                If a token in @token_list does not occur in the 
                dictionary, then it will not be included in the 
                bag-of-words output.
                """
                count = {};

                for tok in token_list:
                        if not isinstance(tok, unicode):
                                tok = unicode(tok, "utf-8");

                        if tok in count:
                                count[tok] += 1;
                        else:
                                count[tok] = 1;

                bow = numpy.zeros(len(self), dtype=numpy.int16);
                
                for tok in count:
                        if tok in self.vocabulary:
                                bow[self.vocabulary[tok]] = count[tok];

                return bow;

        def resize(self, num_tokens=100000):
                """Resize the dictionary to hold only the most frequent tokens.

                Args:
                        @self      : The dictionary object (automatic) 
                        @num_tokens: Maximum number of tokens to keep.

                Return:
                        Nothing

                NOTE
                Tokens are sorted by frequency in terms of the number
                of documents in which they appear (repeated occurrences
                of tokens in a single document do not contribute to the 
                frequency).

                After resizing, tokens will be re-keyed, so that a given 
                token MAY HAVE A DIFFERENT ID before and after calling this
                function. 
                """
                if len(self) > num_tokens:
                        # Sort by number of document appearances 
                        ids = sorted(self.vocabulary, key=self.frequency.get);
                        ids = ids[max_num_tokens:];

                        # Get rid of offending ids 
                        ids = set(ids)
                        for tok, tok_id in self.vocabulary.items():
                                if tok_id in ids:
                                        del self.vocabulary[tok];
                                        del self.frequency[tok_id];

                        # Reassign ids
                        i = 0;
                        for tok in self.vocabulary:
                                self.vocabulary[tok] = i;
                                i += 1;

                        # This is now useless until we get more fancy.
                        self.frequency = {};

class BTL_lda:
        model = None;

        def __init__(self, dtm, **kwargs):
                assert dtm is not None;

                num_topics = kwargs.pop("num_topics", 100);
                num_passes = kwargs.pop("num_passes", 10);

                self.model = lda.LDA(
                        n_topics = num_topics,
                        n_iter   = num_passes
                );

                # Fit the model
                self.model.fit(dtm);

        def theta(self):
                return self.model.doc_topic_;

        def phi(self):
                return self.model.topic_word_;

        def save(self, phi_path=None, theta_path=None):
                numpy.savez(phi_path, self.model.topic_word_);
                numpy.savez(theta_path, self.model.doc_topic_);

        def topic(self, topic_ids):
                """Retreive a topic or list of topics from a topic_id.
                        
                Note that a "topic" is a vector representing a
                discrete probability distribution over the dictionary.
                        
                From lda.py documentation on .components_:
                   
                arraylike (array, shape = [n_topics, n_features]) 
                Point estimate of the topic-word distributions 
                (Phi in literature)

                """
                if hasattr(topic_ids, "__iter__"): 
                        return [self.model.components_[t] for t in topic_ids];
                else:
                        return self.model.components_[topic_ids];

        def find_topics(self, bow, limit=10):
                """Convert a document (bag-of-words) into list of topic_id

                The result will be sorted by probability (high to low).
                """
                topics = self.model.transform(bow)[0];
                topics = [(i, val) for i, val in enumerate(topics)];
                topics = sorted(topics, key=lambda x: x[1], reverse=True);
                topics = topics[:limit];

                return topics;


class BTL_corpus:
        dictionary = None;  # Dictionary (token_id => word mapping)
        dtm        = None;  # Document-term matrix ((doc_id, term_id) => count) 

        loaded = False;

        def __init__(self, tokenizer=None, documents=None):
                if tokenizer is not None:
                        self.tokenizer = tokenizer;
                else:
                        self.tokenizer = BTL_tokenizer(
                                stop_tokens           = True,
                                use_lexical_smoothing = True,        
                                use_stemming          = True,
                                use_pos_tagging       = True,
                        );

                if documents is not None:
                        self._add_documents(documents);

        def save(self, dictionary_path=None, dtm_path=None):
                pickle_it(dictionary_path, self.dictionary.table);
                numpy.savez(dtm_path, self.dtm);

        def load(self, dictionary_path=None, dtm_path=None):
                self.dictionary = BTL_dictionary(unpickle_it(dictionary_path));
                self.dtm        = numpy.load(dtm_path);

                self.loaded = True;

        def _add_documents(self, doc_list):

                # No mutation allowed... for now
                assert self.loaded is False;

                if self.dictionary is None:
                        self.dictionary = BTL_dictionary();

                # Build up the dictionary
                for doc in doc_list:
                        self.dictionary.grow(doc);
                        print("dictionary size:%d" % (len(self.dictionary)));

                self.dictionary.winnow(max_size=30000);

                # Build the DTM
                self.dtm = scipy.sparse.csr_matrix([self.dictionary.bow(d) for d in doc_list]);
                self.dtm.shape = (len(doc_list), len(self.dictionary));

        def lda(self, **kwargs):
                return BTL_lda(self.dtm, **kwargs);


def format_theta(theta_topic_row, limit=10):
        """Convert full theta row (topic) into compact form

        The result will be sorted by probability (high to low).
        """
        topics = theta_topic_row;
        topics = [(i, val) for i, val in enumerate(topics)];
        topics = sorted(topics, key=lambda x: x[1], reverse=True);
        topics = topics[:limit];

        return topics;


def btl_similarity_matrix(theta=None, M_T=10, M_O=10):

        # doc_id => M_T highest-probability topic_ids 
        DOC_TOPICS = {};

        # topic_id => M_O highest-probability doc_ids 
        TOPIC_DOCS = {};

        # For each document in the corpus 
        for doc_id in range(len(theta)): 

                topics = format_theta(theta[doc_id]);

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
                                        print("[WARN] Equal probabilities.");

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

        return T_sim;


