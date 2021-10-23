import os, re, random, json
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from contextlib import contextmanager
from gensim.models import Word2Vec

# tokenizers
from attacut import tokenize # thai
from nltk.tokenize import sent_tokenize # english
from nltk.tokenize import word_tokenize # english

@contextmanager
def time_(name):
    start_time = datetime.now()
    yield
    elapsed_time = datetime.now() - start_time
    print(f'{name} finished in {str(elapsed_time)[:7]}')

def create_corpus_word_pt():
    '''
    prepare corpus_word for pt
    '''
    
    wikicorpus, corpus = 'wiki_th', 'pt'
    
    corpus_words = []

    with open(f'corpus/{wikicorpus}/stopwords.txt') as f:
        stopwords = f.read().strip().split('\n')

    df = pd.read_csv('data/pt/prachathai-67k.csv')
    texts = df['body_text']
    for text in texts:
        lines = text.split('\n')
        sent_text = ''
        for line in lines:
            tokens = [t for t in tokenize(line) if re.match('^[ก-์]+$', t)]
            tokens = [t for t in tokens if t not in stopwords]
            if len(tokens) > 1:
                sent_text += ' '.join(tokens) + '\n' 
        corpus_words += [sent_text.strip()]
        
    random.shuffle(corpus_words)
    print('number of paragraph:', len(corpus_words))

    with open(f'corpus/{corpus}/corpus_word.txt', 'w') as f:
        f.write('\n\n'.join(corpus_words))

def create_corpus_word_wn():
    '''
    prepare corpus_word for wn
    '''
    
    wikicorpus, corpus = 'wiki_th', 'wn'
    
    corpus_words = []

    with open(f'corpus/{wikicorpus}/stopwords.txt') as f:
        stopwords = f.read().strip().split('\n')
        
    df = pd.read_csv('data/wn/wn.csv')
    texts = df['text']
    for text in texts:
        lines = text.split('\n')
        sent_text = ''
        for line in lines:
            tokens = [t for t in tokenize(line) if re.match('^[ก-์]+$', t)]
            tokens = [t for t in tokens if t not in stopwords]
            if len(tokens) > 1:
                sent_text += ' '.join(tokens) + '\n' 
        corpus_words += [sent_text.strip()]
        
    random.shuffle(corpus_words)
    print('number of paragraph:', len(corpus_words))

    with open(f'corpus/{corpus}/corpus_word.txt', 'w') as f:
        f.write('\n\n'.join(corpus_words))

def create_corpus_word_ny():
    '''
    prepare corpus_word for ny
    '''
    
    wikicorpus, corpus = 'wiki_en', 'ny'
    
    corpus_words = []

    with open(f'corpus/{wikicorpus}/stopwords.txt') as f:
        stopwords = f.read().strip().split('\n')
        
    with open('data/ny/nytimes_news_articles.txt') as f:
        obj = f.read()
    articles = obj.split('\n\n')
    articles = [a for a in articles if a[:3]!='URL']
    articles = articles[::3]
    for article in articles:
        try:
            start = re.search('—', article).start() + 2
        except:
            start = 0
        article = article[start:]
        
        sent_text = ''
        
        lines = article.split('\n')
        for line in lines:
            sents = sent_tokenize(line)
            for sent in sents:
                tokens = [t for t in word_tokenize(sent) if re.match('^[a-zA-Z]+$', t)] # english word only
                tokens = [t for t in tokens if t not in stopwords]
                if len(tokens) > 1:
                    sent_text += ' '.join(tokens) + '\n'
        corpus_words += [sent_text.strip()]
        
    random.shuffle(corpus_words)
    print('number of paragraph:', len(corpus_words))

    with open(f'corpus/{corpus}/corpus_word.txt', 'w') as f:
        f.write('\n\n'.join(corpus_words))

def create_corpus_word_ye():
    '''
    prepare corpus_word for ny
    '''
    
    wikicorpus, corpus = 'wiki_en', 'ye'
    
    corpus_words = []

    with open(f'corpus/{wikicorpus}/stopwords.txt') as f:
        stopwords = f.read().strip().split('\n')
        
    df = pd.read_csv('data/ye/yelp.csv')
    texts = df['text']
    texts = texts[::6]
    for text in texts[:100]:
        lines = text.split('\n')
        sent_text = ''
        for line in lines:
            sents = sent_tokenize(line)
            for sent in sents:
                tokens = [t for t in word_tokenize(sent) if re.match('^[a-zA-Z]+$', t)] # english word only
                tokens = [t for t in tokens if t not in stopwords]
                if len(tokens) > 1:
                    sent_text += ' '.join(tokens) + '\n'
        corpus_words += [sent_text.strip()]
        
    random.shuffle(corpus_words)
    print('number of paragraph:', len(corpus_words))

    with open(f'corpus/{corpus}/corpus_word.txt', 'w') as f:
        f.write('\n\n'.join(corpus_words))

def prepare_dir(corpus, type_):
    '''
    remove previous directory if existed,
    and create train and test directories each type of collocation
    '''
    os.system(f'rm corpus/{corpus}/{type_} -r -f')
    os.system(f'mkdir corpus/{corpus}/{type_}')
    os.system(f'mkdir corpus/{corpus}/{type_}/dir_test')
    os.system(f'mkdir corpus/{corpus}/{type_}/dir_train')

def write_train_test(corpus, type_, paras):
    '''
    write data from paras into files that mallet needs
    '''
    prepare_dir(corpus, type_)

    # test
    for i, para in enumerate(paras[:len(paras)//4]): # first 25%
        with open(f'corpus/{corpus}/{type_}/dir_test/{i}.txt', 'w') as f:
            f.write(para.replace('\n', ' '))
    # train
    for i, para in enumerate(paras[len(paras)//4:]): # last 75%
        with open(f'corpus/{corpus}/{type_}/dir_train/{i}.txt', 'w') as f:
            f.write(para.replace('\n', ' '))

def prepare_mallet_data(corpus, types):
    '''
    prepare files that mallet needs
    '''
    for type_ in types:
        prepare_dir(corpus, type_)
        with open(f'corpus/{corpus}/corpus_{type_}.txt') as f:
            paras = f.read().split('\n\n')
        write_train_test(corpus, type_, paras)                                          

def merge_tokens(text, top_bigrams):
    '''
    merge text using top bigrams
    '''
    tokens = text.strip().split()
    merged_text = tokens[0] # text starts with first token

    for i in range(len(tokens)-1):
        bigram = tokens[i]+'\t'+tokens[i+1]
        if bigram in top_bigrams:
            merged_text += '_' + tokens[i+1] # merge
        else:
            merged_text += ' ' + tokens[i+1] # not merge
    
    return merged_text

def create_corpus(corpus, wikicorpus, types):
    '''
    create corpus for collocations
    '''
    for type_ in types:

        corpus_text = ''
        # load bi_to_vocabs
        with open(f'corpus/{wikicorpus}/top_bigrams_{type_}.txt') as f:
            top_bigrams = f.read().split('\n')
            
        # load original text
        with open(f'corpus/{corpus}/corpus_word.txt') as f:
            paras = f.read().split('\n\n')
        
        # merge tokens
        for para in tqdm(paras):
            sents = para.split('\n')
            for sent in sents:
                corpus_text += merge_tokens(sent, top_bigrams) + '\n' # merge tokens
            corpus_text += '\n'
        
        # write files
        with open(f'corpus/{corpus}/corpus_{type_}.txt', 'w') as f:
            f.write(corpus_text.strip())

def prepare_w2v_input(corpus, types):
    '''
    open corpus for each type,
    convert to list of list of tokens, and
    save as input for word2vec
    '''
    input_ = []
    for type_ in types:
        with open(f'corpus/{corpus}/corpus_{type_}.txt') as f:
            paras = f.read().split('\n\n')
            sents = [sent.split() for para in paras for sent in para.split('\n')]
            input_ += sents
    return input_

def train_w2v(corpus, types, vector_size=100, min_count=1, epochs=40):
    input_ = prepare_w2v_input(corpus, types)
    with time_('w2v'):
        model = Word2Vec(input_, vector_size=vector_size, min_count=min_count, epochs=epochs)
        model.save(f'corpus/{corpus}/w2v_model')

def export_doc_tok():
    
    with open('corpus/abv_text.txt') as f:
        abvs = f.read().strip().split('\n')
        abv_text = {abv.split(':')[0]:abv.split(':')[1] for abv in abvs}
    
    text = ''
    corpora = ['ny', 'ye', 'cn', 'dp', 'pt', 'wn']
    for corpus in tqdm(corpora):
        with open(f'corpus/{corpus}/corpus_word.txt') as f:
            obj = f.read().strip()
            n_doc = len(obj.split('\n'))
            n_token = len(obj.replace('\n', ' ').split())
            text += f'{abv_text[corpus]}, {n_doc/1000:.0f}K, {n_token/1000000:.1f}M\n'
    with open('corpus/doctok.csv', 'w') as f:
        f.write(text)


