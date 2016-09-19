# -*- coding: UTF-8 -*-
"""
corpus.py
`````````
Functions and classes for tokenizing texts, constructing a
dictionary, and computing a document-term matrix suitable 
for input to a topic modeller such as LDA.
"""
import sys

# Support locally-installed libraries
sys.path.append("./lib") 

import re       # Regular expressions used in tokenizer
import nltk     # Used for part-of-speech tagging 
import numpy    # Used for various functions 
import scipy    # Used for sparse arrays
import sqlite3  # Part of standard library, used in from_database()
import os       # Used to iterate files in from_directory()

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
# TOKENIZER constants 
###############################################################################

# Ensure that the POS tagging database is available.
# nltk.download("averaged_perceptron_tagger");

# Tell NLTK where it can find the data for Part-of-speech tagging. 
nltk.data.path.append("./extra/nltk_data/");

# English stopword list from Stone, Denis, Kwantes (2010)
STOPWORDS = """
a about above across after afterwards again against all almost alone along 
already also although always am among amongst amoungst amount an and another 
any anyhow anyone anything anyway anywhere are around as at back be became 
because become becomes becoming been before beforehand behind being below 
beside besides between beyond bill both bottom but by call can cannot cant co 
computer con could couldnt cry de describe detail did didn do does doesn doing 
don done down due during each eg eight either eleven else elsewhere empty 
enough etc even ever every everyone everything everywhere except few fifteen
fify fill find fire first five for former formerly forty found four from front 
full further get give go had has hasnt have he hence her here hereafter hereby 
herein hereupon hers herself him himself his how however hundred i ie if in inc 
indeed interest into is it its itself keep last latter latterly least less ltd
just kg km made make many may me meanwhile might mill mine more moreover most 
mostly move much must my myself name namely neither never nevertheless next 
nine no nobody none noone nor not nothing now nowhere of off often on once one 
only onto or other others otherwise our ours ourselves out over own part per
perhaps please put rather re quite rather really regarding same say see seem 
seemed seeming seems serious several she should show side since sincere six 
sixty so some somehow someone something sometime sometimes somewhere still such 
system take ten than that the their them themselves then thence there 
thereafter thereby therefore therein thereupon these they thick thin third this 
those though three through throughout thru thus to together too top toward 
towards twelve twenty two un under until up unless upon us used using various 
very very via was we well were what whatever when whence whenever where 
whereafter whereas whereby wherein whereupon wherever whether which while 
whither who whoever whole whom whose why will with within without would yet you
your yours yourself yourselves
"""
STOPWORDS = [w for w in STOPWORDS.split() if w];

###############################################################################
# Classes 
###############################################################################

class Tokenizer:
        """ 
        To enhance the fit of the topic models, various strategies
        can be used to filter out non-semantic noise from the texts
        being tokenized. 
        
        These are, in order of application:
         
        1.) Lexical filtering 
              a. All sequences of whitespace become a single space
              b. All non-ASCII and punctuation characters are removed
              c. All letters are made lowercase
       
        2.) Part-of-speech (POS) filtering 
              a. Words that are not nouns are removed
        
        3.) Stop word filtering 
              a. Tokens in NLTK English stop word list are removed. 
              b. Tokens in user-provided stop word list are removed.
               
        4.) Stemming
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

        def filter_lexical(self, string):
                """ 
                Remove certain lexical features from the given string

                Arguments:
                        @self  : Tokenizer object (automatic)
                        @string: Input string to be manipulated
                Return: 
                        The smoothed string
                EXAMPLE:
                        "are  you\\nwell--sailor John?" => "are you well sailor john"
                """
                # Replace all whitespace sequences with single space
                string = self.match_whitespace.sub(" ", string);

                # Remove all undesirable characters from string
                string = self.match_disallowed.sub("", string);

                # Convert all letters to lowercase
                string = string.lower();

                return string;

        def filter_part_of_speech(self, tokens):
                """ 
                Remove certain parts of speech from the token list

                Arguments:
                        @self      : Tokenizer object (automatic)
                        @token_list: Input as an array of words 
                Return: 
                        Some subset of the token list
                EXAMPLE:
                        ["are","you","well","sailor","john"] => ["you","sailor","john"]
                """
                tagged = nltk.pos_tag(tokens);

                return [w[0] for w in tagged if w[1][0] == "N"];

        def filter_stop_tokens(self, token_list):
                """ 
                Remove stop tokens from the token list.

                Arguments:
                        @self      : Tokenizer object (automatic)
                        @token_list: List of Unicode tokens 
                Return: 
                        Some subset of the tokenized_string 
                EXAMPLE:
                        ["you","sailor","john"] => ["sailor","john"]
                """
                return [tok for tok in token_list if not tok in self.stop_tokens];

        def filter_stem_tokens(self, token_list): 
                """ 
                Perform stemming on the provided token list

                Arguments:
                        @self      : Tokenizer object (automatic)
                        @token_list: List of Unicode tokens 
                Return: 
                        List of stemmed unicode tokens
                EXAMPLE:
                        ["sailor","john"] => ["sail","john"]
                """
                return [self.stemmer.stem(tok) for tok in token_list];

        def tokenize(self, string):
                """
                Convert a string into a token list

                Arguments:
                        @self  : Tokenizer object (automatic)
                        @string: Input string
                Return: 
                        Processed token list (array of strings)
                """
                if self.do_lex:
                        string = self.filter_lexical(string);

                tokens = string.split();

                if self.do_pos:
                        tokens = self.filter_part_of_speech(tokens);

                if self.do_stop:
                        tokens = self.filter_stop_tokens(tokens);

                if self.do_stem:
                        tokens = self.filter_stem_tokens(tokens);

                return tokens;

        def __init__(self, **kwargs):
                """
                Create a new Tokenizer object

                Arguments:
                        @self  : Tokenizer object (automatic)
                        @kwargs: Associative array of options
                Return: 
                        Nothing 
                Options:
                        (BOOL) use_lexical_smoothing 
                        (BOOL) use_pos_tagging
                        (BOOL) use_stemming 
                        (BOOL)|(ARRAY OF STRING) stop_tokens 
                """
                toks = kwargs.pop("stop_tokens", True);

                self.do_stop = (toks is not False);

                if isinstance(toks, str) and os.path.exists(toks): 
                        self.stop_tokens += [l.rstrip("\n") for l in open(toks)];
                elif isinstance(toks, list):
                        self.stop_tokens += toks;

                self.do_lex  = kwargs.pop("use_lexical_smoothing", True);
                self.do_pos  = kwargs.pop("use_pos_tagging", True);
                self.do_stem = kwargs.pop("use_stemming", True);


class Dictionary:
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

        def save(self, path):
                """
                Save the dictionary as 2-column text

                Arguments:
                        @self: The dictionary object (automatic) 
                        @path: Path to save the dictionary to
                Return: 
                        Nothing
                """
                try:
                        f = open(path, "w");
                except IOError:
                        LOG.info("Can't save dictionary to "+path+" (couldn't open file)");
                else:
                        LOG.info("Saving dictionary ("+str(len(self))+" tokens) to "+path);

                for tok, tok_id in self.vocabulary.items():
                        f.write(str(tok) + " " + str(tok_id) + "\n")

                f.close()

                LOG.info("Done");

        def load(self, path):
                """
                Load the dictionary from a 2-column text file

                Arguments:
                        @self: The dictionary object (automatic) 
                        @path: Path to load the dictionary from 
                Return: 
                        Nothing
                """
                try: 
                        f = open(path, "w");
                except IOError: 
                        LOG.info("Can't load dictionary from "+path+" (file does not exist)");
                else:
                        LOG.info("Loading dictionary from "+path);

                for line in f:
                        (tok, tok_id) = line.split();

                        if not isinstance(tok, unicode):
                                tok = unicode(tok, "utf-8");
                        
                        self.vocabulary[tok] = tok_id; 

                LOG.info("Done (got "+str(len(self))+" tokens)");

                f.close();

        def lookup(self, tok_id): 
                """
                Lookup a token from a token id.

                Arguments:
                        @self  : The dictionary object (automatic) 
                        @tok_id: ID of the token to look up
                Return: 
                        A unicode string (the token), or None. 
                Notes:
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

        def grow(self, token_list, give_bow=False):
                """
                Add tokens to the dictionary from a token list 
                        
                Arguments:
                        @self      : The dictionary object (automatic) 
                        @token_list: List of unicode strings
                        @give_bow  : (BOOL) get bag-of-words from @token_list
                Return: 
                        Nothing, or bag-of-words corresponding to @token_list
                Notes:
                        By convention, the @token_list provided to each call 
                        of .grow() is considered to represent an individual 
                        document. 
                        
                        To count token document frequencies, the dictionary 
                        will treat distinct calls to .grow() as distinct 
                        documents, even if the values of @token_list are 
                        identical, and no new tokens are added to the 
                        dictionary.
                        
                        Likewise, if a single document is provided to the 
                        dictionary across multiple calls to .grow(), it will 
                        not be treated as a single document. 

                        Outside of the calculation of token document 
                        frequencies, there will be no adverse effect.
                """
                if give_bow == True:
                        bow = {};
                else:
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
                                tok_id = self.vocabulary[tok];

                                self.frequency[tok_id] += 1;

                        if give_bow == True:
                                # Construct a bag of words to return 
                                if tok_id in bow:
                                        bow[tok_id] += 1;
                                else:
                                        bow[tok_id] = 1;

                if give_bow == True:
                        return bow;

        def bow(self, token_list):
                """
                Convert a token list into a bag of words 

                Arguments:
                        @self      : The dictionary object (automatic) 
                        @token_list: List of Unicode strings.
                Return: 
                        Numpy array of token frequencies. 

                Note 1:
                        The returned array is of the form

                                [f_0, f_1, f_2, ..., f_n],

                        where n = len(self) is the number of tokens in the 
                        dictionary, and f_i is the number of times that the
                        token having id 'i' (in the dictionary) occurs in 
                        @token_list.
                Note 2:
                        The length of the returned array is equal to the
                        length of the dictionary, even though in practice
                        most values will be 0. 

                Note 3:
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
                """
                Resize the dictionary to hold only the most frequent tokens.

                Arguments:
                        @self      : The dictionary object (automatic) 
                        @num_tokens: Maximum number of tokens to keep.
                Return: 
                        Nothing
                Note: 
                        Tokens are sorted by frequency in terms of the number
                        of documents in which they appear (repeated occurrences
                        of tokens in a single document do not contribute to the 
                        frequency).

                        After resizing, tokens will be re-keyed, so that a 
                        given token MAY HAVE A DIFFERENT ID before and after 
                        calling this function. 
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


class Loader:
        """
        Instantiate a loader object to construct a dictionary and 
        DTM by iteratively adding documents.

        This ensures a reasonable degree of "memory independence"
        -- i.e., your entire list of documents need not be held
        in RAM at once.
        """
        dictionary = None; # Dictionary (token_id => word mapping)
        bow_list   = [];   # Temporary bag-of-word list used when loading corpus

        def __init__(self, max_tokens=None):
                """
                Instantiate a loader object

                Arguments:
                        @self      : Loader object (automatic)
                        @max_tokens: Maximum number of tokens in the dictionary
                Return:
                        Nothing
                """
                self.max_tokens = max_tokens;
                self.dictionary = Dictionary();

                LOG.info("Created loader");


        def add_document(self, token_list):
                """
                Add a document to the loader, growing the dictionary etc.

                Arguments:
                        @self      : Loader object (automatic)
                        @token_list: List of Unicode strings
                Return:
                        Nothing.
                Note:
                        The provided token list is provided to the .grow()
                        method of the dictionary, which returns a bag of
                        words for the token list. This is stored until
                        final computation of the DTM when calling .done().
                """
                self.bow_list.append(self.dictionary.grow(token_list, True));

                LOG.info("Loading document "+str(len(self.bow_list)-1)+" dictionary size: "+str(len(self.dictionary))+" tokens");

        def done(self):
                """
                Complete the loading. 

                Arguments:
                        @self: Loader object (automatic)
                Return:
                        (dictionary, dtm) pair. 
                Note:
                        This method must be called in order to receive 
                        useful output from the loader. After calling
                        .done(), the loader should not be used again.
                """
                if self.max_tokens is not None:
                        # FIXME: 
                        #       It is too late to reduce the vocabulary for 
                        #       two reasons:
                        #       1.) The bags-of-words are already the size 
                        #           they are and you won't be changing them.
                        #       2.) Re-keying the vocabulary table will make 
                        #           the term_ids used in the stored 
                        #           bags-of-words pointless.
                        # SOLUTION: 
                        #       Re-key the bags-of-words you have already,
                        #       according to the re-sized dictionary, and
                        #       discard ones that are no longer present in
                        #       the vocabulary.
                        #
                        #       Is it worth the trouble? The dictionary is
                        #       <100k entries with POS tagging enabled.
                        self.dictionary.resize(self.max_tokens);

                LOG.info("Loader is done.");

                return (self.dictionary, self._compute_dtm());

        def _compute_dtm(self):
                """
                Compute the final DTM

                Arguments:
                        @self      : Loader object (automatic)
                Return:
                        Sparse CSR matrix representing the DTM.
                Note:
                        The reason we compute the matrix at the end is 
                        simply because we don't know the final size of
                        the matrix -- i.e. the number of documents --
                        until .done() is called. We could simply resize
                        it as we go, but I am cautious and there may be 
                        other problems eventually. One I can think of is
                        that calling dictionary.resize() will, among other
                        things, re-key the vocabulary table, so that any
                        term_ids used in the bags-of-words will no longer
                        be accurate. In fact, that is a problem right now.
                """
                LOG.info("Loader computing DTM");
                # 
                # This is the standard efficient construction of a
                # compressed sparse row (CSR, or Yale-format) matrix.
                # If it seems weird, look it up.
                #
                row_pointer = [0]
                col_indices = []
                data        = []

                for bow in self.bow_list:
                        for tok_id, freq in bow.items():
                                col_indices.append(tok_id);
                                data.append(freq);

                        row_pointer.append(len(col_indices));

                return scipy.sparse.csr_matrix(
                        (data, col_indices, row_pointer), 
                        shape = (len(self.bow_list), len(self.dictionary)), 
                        dtype = int
                );


###############################################################################
# Convenience corpus loaders 
###############################################################################

def from_list(tokenize, document_list):
        """
        Create corpus from Python list (array) of Unicode strings.
        Each string in the list represents an individual document.

        Arguments:
                @tokenize     : Function to tokenize string => list of string
                @document_list: List (array) of Unicode strings
        Return:
                @dictionary: corpus.Dictionary object
                @dtm       : scipy.sparse.csr_matrix (document-term matrix)
        """
        LOG.info("Loading corpus from list of Unicode strings");
        loader = Loader();

        for doc in document_list:
                loader.add_document(tokenize(doc));

        return loader.done();

def from_directory(tokenize, path):
        """
        Create corpus from a directory of flat text files. 
        The content of each file represents an individual document.

        Arguments:
                @tokenize: Function to tokenize string => list of string
                @path    : Directory path (string)
        Return:
                @dictionary: corpus.Dictionary object
                @dtm       : scipy.sparse.csr_matrix (document-term matrix)
        """
        LOG.info("Loading corpus from directory: "+path);
        loader = Loader();

        for name in os.listdir(path): 
                loader.add_document(tokenize(open(path+"/"+name).read()));

        return loader.done();

def from_database(tokenize, path=None, query=None):
        """
        Create corpus from a SQLite3 database and database query. 
        Each row of the query result represents an individual document.

        Arguments:
                @tokenize: Function to tokenize string => list of string
                @path    : Path to the SQLite3 database file.
                @query   : Query to execute on the database.
        Return:
                @dictionary: corpus.Dictionary object
                @dtm       : scipy.sparse.csr_matrix (document-term matrix)
        Note:
                When run on the database at @path, @query should result
                in a single column of results. If there is more than one
                column, only the first will be used.
        """
        LOG.info("Loading corpus from database: "+path);
        loader = Loader();

        try: 
                conn = sqlite3.connect(path);
        except sqlite3.Error as e:
                LOG.error("SQLITE3 error: ", e.args[0]);

        # For processing non-ASCII characters
        conn.text_factory = bytes; 

        db = conn.cursor(); 
        db.execute(query);

        for row in db:
                loader.add_document(tokenize(row[0]));

        return loader.done();
