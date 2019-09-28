import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("./dataset/sp.et.model")
message = open("et.enc.txt", "r", encoding="utf8").read()
message_decoded = sp.DecodePieces(message.split())

print(message_decoded)

file = open("et.txt", "w", encoding="utf-8")
file.write(message_decoded)
file.close()