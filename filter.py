import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]


def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')


def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


def main(input, predictions):

    tmp = [pred.replace("'","").split(". ") for pred in predictions ]

    result = []
    for r in tmp:
      result += r

    sim = []
    for i in range(len(tmp)):
      for s in tmp[i]:
        sim.append(cosine_sim(input[i], s))

    del_index = [c for c,i in enumerate(sim) if i<0.2]

    try :
      for i in del_index:
        del result[i]
    except:
      pass

    result = [x.strip() for x in result]
    if result == "":
      result = input

    return result
