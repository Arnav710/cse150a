# AI Medical Chatbot

## Explain what your AI agent does in terms of PEAS. What is the "world" like? 

## What kind of agent is it? Goal based? Utility based? etc. 

## Describe how your agent is set up and where it fits in probabilistic modeling

## Data Exploration (link)

## Data Preprocessing ([link](https://github.com/Arnav710/cse150a/blob/main/data_preprocessing.ipynb))

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

## Models and Evaluation ([link](https://github.com/Arnav710/cse150a/blob/main/models.ipynb))

Before training the model, we split our dataset into train, test and validation components.

After performing the split, their sizes were as follows:
```
Number of samples in train set: 208097
Number of samples in validation set: 23122
Number of samples in test set: 25691
```

The following descibes the models we built and their performance:

#### `ProbabilityBasedAgent`

The probability based agent constructs the vocabulary by looking at all the 
questions in the dataset. 

When it is given a user query, iterates over all records and tries to find the 
most similar question by computing `P(Query | i-th sentence)`. Laplace smoothening
is done to avoid 0 probabilities.

It then tries to maximize this probability and outputs the corresponding response.

```
class ProbabilityBasedAgent:
    
	def __init__(self, questions, responses):
		self.questions = questions
		self.responses = responses
		self.question_sets = []
		self.vocab = None

	def get_vocab(self):
		vocab = set()
		for question in self.questions:
			for word in question.split():
				vocab.add(word)
		return list(vocab)

	def prob_query_given_sentence(self, query, sentence_lst, alpha=1):
		query_lst = query.split()
		match = 0
		
		for token in query_lst:
			if token in sentence_lst:
				match += 1
		
		# Apply Laplace smoothing
		numerator = match + alpha
		denominator = len(query_lst) + alpha
		
		p = numerator / denominator
		return p

	def train(self):
		self.vocab = self.get_vocab()

		for question in self.questions:
			self.question_sets.append(set(question.split()))

	def find_closest_answer(self, query, k):
		
		probabilities_match = []
		for i in range(len(self.questions)):
			prob = self.prob_query_given_sentence(query, self.question_sets[i])
			probabilities_match.append((prob, self.questions[i], self.responses[i]))

		probabilities_match.sort(reverse=True)

		return probabilities_match[:k]
```


#### `SimilarityBasedAgent`


The similarity based agent constructs the vocabulary by looking at all the 
questions in the dataset. It then converts the sentence into vectors using
a using the Bag of Words approach.

When it is given a user query, iterates over all records and tries to compute
the cosine similarity between th user query vector and the vector associated
with each of the questions in the dataset.

It then tries to maximize this similarity.

```
class SimilarityBasedAgent:
    
	def __init__(self, questions, responses):
		self.questions = questions
		self.responses = responses
		self.vocab = None
		self.questions_vectors = None

	def get_vocab(self):
		vocab = set()
		for question in self.questions:
			for word in question.split():
				vocab.add(word)
		return list(vocab)
	
	def bag_of_words(self, question):
		vec = []
		for token in self.vocab:
			vec.append(question.count(token))
		return vec

	def train(self):
		self.vocab = self.get_vocab()
		vectors = []
		for question in self.questions:
			vectors.append(self.bag_of_words(question))
		self.questions_vectors = vectors

	def find_closest_answer(self, query, k):
		user_query_vector = self.bag_of_words(query)
		
		similarities = []
		for i in range(len(self.questions)):
			sim = cosine_similarity(user_query_vector, self.questions_vectors[i])
			similarities.append((sim, self.questions[i], self.responses[i]))

		similarities.sort(reverse=True)

		return similarities[:k]
```

## Evaluation


For every test sample, the overlap between the response returned by the model
and the origianl response in the test set is calculated by looking at their intersection
and then normalizing it.

Performance on test set:
Probability based model: 0.225
Cosine similarity based model: 0.25 

Although these scores may seem low, considering that the model does not have generative
capabilities, they show that there was a decent overlap in the answer produced by the model.
Cosine similarity produced a better result relative to the probability based model showing
that the Bag Of Word feature representations may have been helpful

Using this metric, the performance on the train set was 1 for both models since both
were able to find the exact responses in the dataset.

So, both models were overfitting to the train set. It is important to note this this metric
is biased towards the training samples and we will look for be

## Conclusion section: 
It can be seen that the cosine similarity model performs slightly better than the probability based one.
One reason for this is that converting words into feature vectors using Bag Of Words helps
These models will serve as useful baselines/benchmarks for the models we build for the upcoming milestones.

In the future models we build we plan on exploring/ looking into following:
- Using Hidden Markov Models to perform named entity recognition. This should help focus on certain key parts of the sentence
- Ensembling different kinds of models to see the effect on performance
- Experimenting with different embeddings for the feature vectors that better capture semantic meaning
- Looking for better evaluation metric 
