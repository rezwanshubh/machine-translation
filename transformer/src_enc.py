import sentencepiece as spm

def encodeAndSaveAsText(message):
    sp = spm.SentencePieceProcessor()
    sp.Load("./dataset/sp.en.model")
    encoded_text = sp.EncodeAsPieces(message)
    file = open("en.enc.txt", "w", encoding="utf-8")
    file.write(' '.join(encoded_text))
    file.close()



try:
    message = open("en.txt", "r").read()
    #message = "Weather is sunny today. Let's go out for a walk."
    print('Text: ' + message);
    encodeAndSaveAsText(message)
except:
    pass




