import nltk

domain= 'domain-movie-subtitle'

reference = open(domain + "/et.txt", "r", encoding="utf8").read()
hypothesis = open(domain + "/transformer_et.txt", "r", encoding="utf8").read()
#there may be several references
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
print(BLEUscore)