import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams

class TrigramProbabilityFinder:
	"""
		Given training and held out corpora, this class will
		calculate unigram, bigram, and trigram probabilities
		from the training corpus then use the held-out corpus
		to calculate constants which can be used for finding
		linearly interpolated trigram probability.
	"""
	
	SOS = "<s>"
	EOS = "</s>"
	
	def __init__(self, train_file, held_out_file):
		"""
			This constructor initializes the instance variables, i.e. the
			dictionaries that will store the ngram counts and the lambda values.
			It then reads the head and tail files passed to it and counts uni,
			bi, and trigrams. Because it processes so much data instantiating
			this class will take a few seconds.
		"""
		
		self.unigram_counts = {}
		self.bigram_counts = {}
		self.trigram_counts = {}
		
		self.lambda_norm_1 = ""
		self.lambda_norm_2 = ""
		self.lambda_norm_3 = ""
		
		try:
			train_data = open(train_file,'r')
			train_text = train_data.read()
			train_data.close()
		except(FileNotFoundError):
			sys.exit("Couldn't read training file")
		
		try:
			held_out_data = open(held_out_file,'r')
			held_out_text = held_out_data.read()
			held_out_data.close()
		except(FileNotFoundError):
			sys.exit("Couldn't read held-out file")
		
		self.prepped_train_text = self.add_BOS_EOS(train_text)
		self.prepped_held_out = self.add_BOS_EOS(held_out_text)
		
		self.calculate_unigrams(self.prepped_train_text)
		self.calculate_bigrams(self.prepped_train_text)
		self.calculate_trigrams(self.prepped_train_text)
		
		self.calculate_lambdas(self.prepped_held_out)
		
	def add_BOS_EOS(self, text):
		"""
			given a string, returns a list of word tokens
			which include BOS (<s>) and EOS tags (</s>)
		"""
	
		result = []
		text_sentences = sent_tokenize(text)
		for sentence in text_sentences:
			words = word_tokenize(sentence)
			words.insert(0, self.SOS)
			words.append(self.EOS)
			for word in words:
				result.append(word)
		return result
		
	def calculate_unigrams(self, tokens):
		"""
			Given tokenized text in list form organizes it into a dictionary of types
			where each key is a type and each value is that type's count in the training
			corpus.
		"""
		
		for word in tokens:
			if word in self.unigram_counts:
				self.unigram_counts[word] += 1
			else:
				self.unigram_counts[word] = 1
		result_file = open('unigram_freqs.txt','w')
		total = len(tokens)
		total_unigrams = len(self.unigram_counts)
		for type in self.unigram_counts:
			result_file.write(type)
			result_file.write(" : ")
			result_file.write(str(self.unigram_counts[type]))
			result_file.write("\n")
		result_file.close()
		print('Unique unigrams in training set: {}'.format(total_unigrams))
		print('Total unigram count in training set: {}'.format(total))
		
	def calculate_bigrams(self, tokens):
		"""
			Given tokenized text in list form organizes it into a dictionary of bigrams
			where each key is a bigram tuple and each value is that bigram's count in the
			training corpus.
		"""
		bigram = list(ngrams(tokens, 2))
		total_bigrams = 0
		for bg in bigram:
			if bg[0] != self.EOS: #don't count sentence crossing bigrams, i.e. ("</s>", "<s>")
				if bg in self.bigram_counts:
					self.bigram_counts[bg] += 1
				else:
					self.bigram_counts[bg] = 1
				total_bigrams += 1
		result_file = open('bigram_freqs.txt','w')
		total = len(self.bigram_counts)
		for bg, count in self.bigram_counts.items():
			result_file.write(str(bg))
			result_file.write(" : ")
			result_file.write(str(count))
			result_file.write("\n")
		result_file.close()
		print('Unique bigrams in training set: {}'.format(total))
		print('Total bigram count in training set: {}'.format(total_bigrams))
		
	def calculate_trigrams(self, tokens):
		"""
			Given tokenized text in list form organizes it into a dictionary of trigrams
			where each key is a trigram tuple and each value is that trigram's count in the
			training corpus.
		"""
		trigram = list(ngrams(tokens, 3))
		total_trigrams = 0
		for tg in trigram:
			if tg[0] != self.EOS and tg[1] != self.EOS: #don't count sentence crossing trigrams, i.e. ("</s>", "<s>", "someword") or (".", "</s>", "<s>")
				if tg in self.trigram_counts:
					self.trigram_counts[tg] += 1
				else:
					self.trigram_counts[tg] = 1
				total_trigrams += 1
		result_file = open('trigram_freqs.txt','w')
		total = len(self.trigram_counts)
		for tg, count in self.trigram_counts.items():
			result_file.write(str(tg))
			result_file.write(" : ")
			result_file.write(str(count))
			result_file.write("\n")
		result_file.close()
		print('Unique trigrams in training set: {}'.format(total))
		print('Total trigram count in training set: {}'.format(total_trigrams))
		
	def calculate_lambdas(self, held_out):
		"""
			Uses a standard method to find constants which will later be applied
			to linear interpolation algorithm.
		"""
		lambda_3 = 0
		lambda_2 = 0
		lambda_1 = 0
		
		trigrams = list(ngrams(held_out, 3))
		
		for tg in trigrams:
			if tg in self.trigram_counts:
				lambda_3 += 1
			elif tg[1:3] in self.bigram_counts:
				lambda_2 += 1
			elif tg[2] in self.unigram_counts:
				lambda_1 += 1
			
		#normalize lambdas
		sum = lambda_3 + lambda_2 + lambda_1
		self.lambda_norm_3 = lambda_3 / sum
		self.lambda_norm_2 = lambda_2 / sum
		self.lambda_norm_1 = lambda_1 / sum
	
		print("Lambdas: Lambda3: {}, Lambda2: {}, Lambda1: {}".format(lambda_3, lambda_2, lambda_1))
		print("Normalized: Lambda3: {}, Lambda2: {}, Lambda1: {}".format(self.lambda_norm_3, self.lambda_norm_2, self.lambda_norm_1))
		
	def find_probability(self, val):
		"""
			This method finds the probability of a passed string using the Markov assumption:
			
			P(<s> i want English food </s>) = P(want|<s> i)P(English|i want)P(food|want English)P(</s>|English food)
			
			Zero probability tri and bigrams are dealt with using the Katz backoff method. After this,
			this method finds the linearly interpolated probability using lambdas calculated in calculate_lambdas().
		"""
		tokens = self.add_BOS_EOS(val)
		trigrams = list(ngrams(tokens, 3))
		total_prob = 1
		total_inter_prob = 1
		for trigram in trigrams:
			C_tri = self.trigram_counts.get(trigram, 0)
			C_bi = self.bigram_counts.get(trigram[0:2], 0)
			C_uni = self.unigram_counts.get(trigram[0], 0)
			print(trigram)
			print(self.trigram_counts.get(trigram))
			
			#use Katz backoff for 0 probabalities e.g. P(English|i want)
			if C_tri == 0:
				if C_bi == 0:
					C_bi = 1 #avoid dividing by zero in interpolated prob
					prob = C_uni / len(self.unigram_counts)
				else:
					prob = C_bi / C_uni
			else:
				prob = C_tri / C_bi
				
			total_prob *= prob
			
			#if unigram count is zero we just substitute bigram probability with zero
			#else unkown words will throw a divide by 0 error
			#it wasn't part of the assignment to handle these cases so unkown words will just get
			#a zero probability, but shouldn't throw an error
			if C_uni == 0:
				inter_prob = self.lambda_norm_3 * (C_tri / C_bi) + self.lambda_norm_2 * 0 + self.lambda_norm_1 * (C_uni / len(self.unigram_counts))
			else:
				inter_prob = self.lambda_norm_3 * (C_tri / C_bi) + self.lambda_norm_2 * (C_bi / C_uni) + self.lambda_norm_1 * (C_uni / len(self.unigram_counts))
			
			total_inter_prob *= inter_prob
			
		print("Normal trigram prob: {}".format(total_prob))
		print("Interpolated trigram prob: {}".format(total_inter_prob))
		
if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Enter the names of your training corpus and held out corpus files at parameters when running this module.")
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	tri_prob_finder = TrigramProbabilityFinder(train_file, test_file)
	tri_prob_finder.find_probability("I want English food")