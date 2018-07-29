from gensim.models import word2vec

raw_sentences = ['the quick from fox jump over lazy dogs', 'yoyo you home now to sleep']

sentences = [s.split() for s in raw_sentences]
print(sentences)

model = word2vec.Word2Vec(sentences, min_count=1)
simil = model.similarity('dogs', 'you')
print(simil)