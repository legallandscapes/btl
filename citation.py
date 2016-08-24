# -*- coding: UTF-8 -*-
import sqlite3  # it's in the standard library
import json

def pickle_it(name, obj):
        with open(name, "wb") as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def unpickle_it(name):
        with open(name, "rb") as f:
                return pickle.load(f)


def btl_citation_matrix():

        # Open a connection to the provided database file
        try: 
                conn = sqlite3.connect("/home/legal_landscapes/public_html/beta/usc.db");
        except sqlite3.Error as e:
                print "SQLITE3 error: ", e.args[0];

        conn.text_factory = bytes; # Process non-ASCII characters

        db = conn.cursor(); 
        db.execute("""
                SELECT n0.rowid, n1.rowid 
                FROM edges
                JOIN nodes as n0
                        ON source_url = n0.url
                JOIN nodes as n1
                        ON target_url = n1.url
        """);

        citation = {};

        for row in db:
                if row[0] in citation:
                        citation[row[0]][row[1]] = 1;
                else:
                        citation[row[0]] = dict([(row[1], 1)]);

        return citation;



cite = btl_citation_matrix();


nodes = [];
links = [];

already_added = {};

for source_id, targets in cite.iteritems():
        nodes.append(dict([("id", source_id), ("group", "0")]));

        for target_id, etc in targets.iteritems():
                if target_id not in cite and target_id not in already_added:
                        nodes.append(dict([("id", target_id), ("group", "0")]));
                        already_added[target_id] = True;

                links.append(dict([("source", source_id), ("target", target_id), ("weight", 1)]));


json_string = json.dumps(dict([("nodes", nodes), ("links", links)]));

print(json_string);
                
#pickle_it("usc.cite", cite);


