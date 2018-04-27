# NLP Programming Assignment #3
# NaiveBayes
# 2012

#
# The area for you to implement is marked with TODO!
# Generally, you should not need to touch things *not* marked TODO
#
# Remember that when you submit your code, it is not run from the command line
# and your main() will *not* be run. To be safest, restrict your changes to
# addExample() and classify() and anything you further invoke from there.
#

# Command line arguments for calling negation, boolean and stopword
# boolean needs to be added in the init of naiveBayes
# 

from collections import defaultdict
import sys
import getopt
import os
import math

class NaiveBayes:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.stopList = set(self.readFile('../data/english.stop'))
    self.numFolds = 10

    # Using the defaultdict data structure to store pos/neg words and their observed frequencies. 
    self.pos_dict = defaultdict(int)
    self.neg_dict = defaultdict(int)

    #Option to negate words.
    self.NEGATION = False

    #Option for boolean
    self.BOOLEAN  = False

  #############################################################################
  # TODO TODO TODO TODO TODO

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    pos_count = 0 
    neg_count = 0 

    # Determining the length of the overall vocabulary set (no duplicates).  
    all_words = list(self.pos_dict.keys()) + list(self.neg_dict.keys())
    unique_words = set(all_words)
    V = len(unique_words) 

    # Initial positive and negative counts.  
    for val in self.pos_dict.values():
      pos_count += val 

    for val in self.neg_dict.values():
      neg_count += val 

    # Using logarithmic transformations to get probabilities; these are the initial yes or no probabilities. 
    p_pos = math.log(pos_count / (pos_count + neg_count))
    p_neg = math.log(neg_count / (pos_count +neg_count))

    posScore = math.log(0.5)
    negScore = math.log(0.5)

    #For the new word, use the mutinomial naive baye's formula. Log transformation since log(xy) = logx + logy. 
    for w in words:
      p_pos += math.log(int(self.pos_dict[w]+1) / (pos_count + abs(V)))
      p_neg += math.log(int((self.neg_dict[w]+1)) / (neg_count + abs(V)))

    # The higher probability determines the classification returned. 
    if p_pos > p_neg:
      return 'pos'
    elif p_neg > p_pos:
      return 'neg'

  def isBool(self, words):
    
    if self.BOOLEAN == True:
      words = set(words)

    return words


  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier
     * in the NaiveBayes class.
     * Returns nothing
    """
    # This makes it true/false

    words = self.isBool(words)

    for word in words:
      if klass == 'pos':
        self.pos_dict[word] += 1

      elif klass == 'neg':
        self.neg_dict[word] += 1

  def filterStopWords(self, words):
    """
    * TODO
    * Filters stop words found in self.stopList.
    """

    # print (words)
    for word in words:
      if word in self.stopList:
        words.remove(word)

    return words

  def negation(self, words):
    negation_words = ['Never', 'Not', 'never', 'not']
    punctuation_stop = ['.','?','!']
    punctuation_ignore = [' ', ',', ';', ':', '(', ')', '{', '}', '[', ']']

    for i in range (len(words)):

        # If the previous word is in the negation list, add NOT_ to this current word. 
        if words[i-1] in negation_words and words[i-1] not in punctuation_stop and words[i] not in punctuation_ignore and words[i] not in punctuation_stop:
          words[i] = 'NOT_' + words[i]

        # If the previous word ends with "n't", then add NOT_ to it. 
        if len(words[i-1]) > 3 and words[i-1][-3] == 'n' and words[i-1][-1] == 't' and words[i-1] not in punctuation_stop and words[i] not in punctuation_ignore and words[i] not in punctuation_stop:
          words[i] = 'NOT_' + words[i]

        # If the the previous word already starts with "NOT_", then add NOT_ to it, unless it's punctuation. 
        if len(words[i-1]) >=3 and words[i-1][0] == 'N' and words[i-1][1] == 'O' and words[i-1][2] == 'T':
          if words[i] not in punctuation_stop and words[i] not in punctuation_ignore:
            words[i] = 'NOT_' + words[i]

        # If the previous word was a non stoping punctuation, then check the word before that for NOT_. If it has it, then add to the current word. 
        if words[i-1] in punctuation_ignore:
          if words[i-2] in negation_words or len(words[i-2]) >=3 and words[i-2][0] == 'N' and words[i-2][1] == 'O' and words[i-2][2] == 'T':
            words[i] = 'NOT_' + words[i]

    return words


  # TODO TODO TODO TODO TODO
  #############################################################################


  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here,
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents))
    return result


  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()


  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)

  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = []
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits


  def test(self, split):
    """Returns a list of labels for split.test."""
    labels = []
    for example in split.test:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      guess = self.classify(words)
      labels.append(guess)
    return labels

  def buildSplits(self, args):
    """Builds the splits for training/testing"""
    trainData = []
    testData = []
    splits = []
    trainDir = args[0]
    if len(args) == 1:
      print('[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir))

      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fold in range(0, self.numFolds):
        split = self.TrainSplit()
        for fileName in posTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
          example.klass = 'pos'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        for fileName in negTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
          example.klass = 'neg'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        splits.append(split)
    elif len(args) == 2:
      split = self.TrainSplit()
      testDir = args[1]
      print('[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir))
      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        split.train.append(example)

      posTestFileNames = os.listdir('%s/pos/' % testDir)
      negTestFileNames = os.listdir('%s/neg/' % testDir)
      for fileName in posTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (testDir, fileName))
        example.klass = 'pos'
        split.test.append(example)
      for fileName in negTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (testDir, fileName))
        example.klass = 'neg'
        split.test.append(example)
      splits.append(split)
    return splits

def main():
  nb = NaiveBayes()

  # default parameters: no stop word filtering, and
  # training/testing on ../data/imdb1
  if len(sys.argv) < 2:
      options = [('','')]
      args = ['../data/imdb1/']
  else:
      # (options, args) = getopt.getopt(sys.argv[1:], 'f')
      options = []
      args = []
      args.append(sys.argv[-1])
      flags = sys.argv[1:-1]

      for flag in flags:
        options.append((flag, ''))

  # python3 NaiveBayes.py -{FLAG_NAME} ../data/imdb1

  if ('-f', '') in options:
    nb.FILTER_STOP_WORDS = True

  if ('-n', '') in options:
    nb.NEGATION = True

  if ('-b', '') in options:
    nb.BOOLEAN = True

  splits = nb.buildSplits(args)
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    accuracy = 0.0
    for example in split.train:
      words = example.words
      if nb.FILTER_STOP_WORDS:
        words =  classifier.filterStopWords(words)
      classifier.addExample(example.klass, words)
      if nb.NEGATION:
        words = classifier.negation(words)
      classifier.addExample(example.klass, words)

    for example in split.test:
      words = example.words
      if nb.FILTER_STOP_WORDS:
        words =  classifier.filterStopWords(words)
      if nb.NEGATION:
        words = classifier.negation(words)
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy))
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print('[INFO]\tAccuracy: %f' % avgAccuracy)

if __name__ == "__main__":
    main()
