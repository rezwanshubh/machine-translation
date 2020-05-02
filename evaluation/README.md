# Project Title

BLEU score evaluation based on different translation model and domain.  

## Description

BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is" – this is the central idea behind BLEU. BLEU was one of the first metrics to claim a high correlation with human judgments of quality and remains one of the most popular automated and inexpensive metrics.

Machine translation evaluation using BLEU is not perfect, still, it's defacto standard in this field. Here we evaluated models in two different domains (movie-subtitle (general), EU-law) when the translation model was developed based on movie-subtitles (general).


## Getting Started

### Dependencies

* Python 3.x

### Executing program

* To evaluate BLEU score in a particular domain.

```
import nltk

domain= 'domain-movie-subtitle'

reference = open(domain + "/et.txt", "r", encoding="utf8").read()
hypothesis = open(domain + "/transformer_et.txt", "r", encoding="utf8").read()
#there may be several references
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
print(BLEUscore)
```

## License

This project is licensed under the MIT License.

## Acknowledgments

Inspiration, code snippets, etc.
* [A Gentle Introduction to Calculating the BLEU Score for Text in Python](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)
* [Natural Language Toolkit](https://www.nltk.org/)
* [BLEU in wikipedia](https://en.wikipedia.org/wiki/BLEU)