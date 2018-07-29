from gensim.models import Word2Vec

en_wiki_word2vec_model = Word2Vec.load('wiki.zh.small.model')

testwords = ['男生','主席','人工智能','手机','中国','美女','百度','佛教']
for i in range(len(testwords)):
    res = en_wiki_word2vec_model.most_similar(testwords[i])
    print (testwords[i])
    print (res)