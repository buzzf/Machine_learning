import pandas as pd
import numpy as np
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
from scipy.misc import imread



df_news = pd.read_table('./data/val.txt',names=['category','theme','URL','content'],encoding='utf-8')
df_news = df_news.dropna()

content = df_news.content.values.tolist()
# print(content[1000])

content_S = []
for line in content:
    current_segment = jieba.lcut(line)
    if len(current_segment) > 1 and current_segment != '\r\n': #换行符
        content_S.append(current_segment)

# print(content_S[1000])

df_content=pd.DataFrame({'content_S':content_S})

stopwords=pd.read_csv("stopwords.txt",index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8')
# print(stopwords.head(20))

def drop_stopwords(contents,stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean,all_words
    #print (contents_clean)
        

contents = df_content.content_S.values.tolist()    
stopwords = stopwords.stopword.values.tolist()
contents_clean,all_words = drop_stopwords(contents,stopwords)
df_all_words=pd.DataFrame({'all_words':all_words})
# print(df_all_words.head())

words_count=df_all_words.groupby(by=['all_words'])['all_words'].agg({"count":np.size})
words_count=words_count.reset_index().sort_values(by=["count"],ascending=False)
# print(words_count.head())

matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

back_mask = imread('R.jpeg')

wordcloud=WordCloud(font_path="./data/simhei.ttf",background_color="white",max_font_size=80,mask=back_mask)
word_frequence = {x[0]:x[1] for x in words_count.head(100).values}
wordcloud=wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
plt.axis('off')
# plt.show()

wordcloud.to_file('news.png')