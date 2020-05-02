import nltk

domain= 'domain-movie-subtitle'

reference = open(domain + "/et.txt", "r", encoding="utf8").read()
hypothesis = open(domain + "/google_et.txt", "r", encoding="utf8").read()
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
print(BLEUscore)

reference = open(domain + "/et.txt", "r", encoding="utf8").read()
hypothesis = open(domain + "/rnmt_plus_et.txt", "r", encoding="utf8").read()
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
print(BLEUscore)

reference = open(domain + "/et.txt", "r", encoding="utf8").read()
hypothesis = open(domain + "/sequence_et.txt", "r", encoding="utf8").read()
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
print(BLEUscore)

reference = open(domain + "/et.txt", "r", encoding="utf8").read()
hypothesis = open(domain + "/transformer_et.txt", "r", encoding="utf8").read()
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
print(BLEUscore)