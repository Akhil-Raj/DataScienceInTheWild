import sys
from collections import defaultdict
 
from pyspark import SparkContext, SparkConf

def processLine(words):
    return [''.join([char for char in words[i].lower() if 97 <= ord(char) <= 122]) + " " + ''.join([char for char in words[i + 1].lower() if 97 <= ord(char) <= 122]) for i in range(len(words) - 1)]

conf = SparkConf()
sc = SparkContext(conf=conf)
bigrams = sc.textFile(sys.argv[1]).flatMap(lambda line: processLine(line.split(" ")))
words = sc.textFile(sys.argv[1]).flatMap(lambda line: line.split(" "))
wordsCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b)
bigramsCounts = bigrams.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b)
wordsWithBigramsCounts = bigramsCounts.map(lambda bigramCount : (bigramCount[0].split(" ")[0], bigramCount))
wordsWithBigramsCountsAndConditionalFrequencyDistribution = wordsWithBigramsCounts.join(wordsCounts)
bigramsWithCountsAndConditionalFrequencyDistribution = wordsWithBigramsCountsAndConditionalFrequencyDistribution.map(lambda ele : (ele[1][0][0], (ele[1][0][1], ele[1][0][1] / ele[1][1])))
#print("bigramsWithConditionalFrequencyDistribution ", bigramsWithConditionalFrequencyDistribution.collect())
bigramsWithCountsAndConditionalFrequencyDistribution.coalesce(1, shuffle=True).saveAsTextFile(sys.argv[2])
sc.stop()

