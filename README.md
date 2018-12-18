# RWOV

Currently, we do not have an example dataset for this software. We cannot release the data used in the paper due to privacy concerns. However, the setup is relatively simple. Please contact us with any questions.

```python
# -*- coding: utf-8 -*-

from RWOV import RWOV

# specify some words to exclude
EXCLUDE = ['the', 'are', 'of', 'as', 'is', 'and', 'or']

# create a list of aliases for the variable of interest
ALIASES = ['er', 'estrogen', 'estrogen receptor']

# specify the number of top words to vectorize
NUM_TOP_WORDS = 30

# this will be removed in a future release, just leave at 0
DEFAULT_REL_LOC = 0

# Get the shortest aliases for the VOI.
voi = RWOV.get_shortest_alias(ALIASES)

# hotspot_rows should be a list containing blocks of text, each containing the
# VOI or variable of interest. voi_rows will contain a list of lists, with
# each list corresponding to one entry in hotspot rows but containing a list
# of stemmed words. all_words is a list of all the words found.
[voi_rows, all_words] = RWV.get_voi_rows(voi, hotspot_rows, ALIASES, EXCLUDE)

# Find the most frequently occuring words.
top_words = RWV.get_top_words(all_words, NUM_TOP_WORDS)

# Create vectors representing the original entries in hotspot rows,
# now vectorized.
word_vectors = RWV.get_word_vectors(voi, voi_rows, top_words, DEFAULT_REL_LOC)
```

A pre-print of the relevant paper is here:
https://arxiv.org/abs/1812.02627
