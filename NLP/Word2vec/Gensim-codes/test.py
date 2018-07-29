import codecs,sys


f=codecs.open('../wiki_data/zhwiki_bj_small_after_jiebaCut.txt','r',encoding="utf8")
# f=codecs.open('../wiki_data/wiki.zh.jian.text','r',encoding="utf8")
line=f.readline()
print(line)