# AI Medical Chatbot

### Explain what your AI agent does in terms of PEAS. What is the "world" like? 

### What kind of agent is it? Goal based? Utility based? etc. 

### Describe how your agent is set up and where it fits in probabilistic modeling

### Data Exploration (link)

### Data Preprocessing (link)

Since all three columns in our dataset involved text data, text related preprocessing
had to be done. The data preprocessing pipeline consisted of the following steps:
- Renaming columns
- Normalizing text to lowercase
- Removing non-alphanumeric characters from the text
- Removing stopwords
- Applying stemming to retain the root words

##### Normalizing text to lowercase
This helps limiting the vocabulary size and thus reducing the dimensionality/length
of the feature vectors when converting text to vector representations. Further, 
this preserves the semantic meaning since being upper case does not change the meaning
of the word

#### Removing non-alphanumeric characters from the text
Symbols do not carry too much semantic meaning. Getting rid of them can reduce the
noise in the data.

#### Removing stopwords
Stopwords help establish the flow of the sentence but do not add too much meaning
to it. They appear very commonly and some examples include "the", "and", "to", "in", "by".
Removing these words helps us focus on the most important aspect of the sentence.

#### Stemming
Stemming allows us to retain only the root words by removing any prefixes or suffixes
that may be associated with the word. This helps limit the size of the vocabulary by
sort of grouping different forms of the same words.


### Train your first model (link)

### Evaluate your model

### Conclusion section: 
What is the conclusion of your 1st model? What can be done to possibly improve it?