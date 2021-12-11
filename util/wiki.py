import re, os, nltk

from bs4 import BeautifulSoup
from tqdm import tqdm

# custom
from util.util import time_

# tokenizers
from attacut import tokenize # thai
from nltk.tokenize import sent_tokenize # english
from nltk.tokenize import word_tokenize # english

def create_wikidump_text(dir_):
    '''
    from the directory of extracted wikidump, create a text file
    '''
    texts = []    

    dirs = os.listdir(f'data/wikidump/{dir_}')
    for d in tqdm(dirs):
        fns = os.listdir(f'data/wikidump/{dir_}/{d}')
        for fn in fns:
            with open(f'data/wikidump/{dir_}/{d}/{fn}') as f:
                contents = f.read()
                soup = BeautifulSoup(contents, 'lxml')
                finds = soup.find_all('doc')
                for find in finds:
                    text = find.text.strip()
                    text = re.sub('\n+', '\n', text)
                    texts += text.split('\n')
                    
    with open(f'data/wikidump/{dir_}_text.txt', 'w') as f:
        f.write('\n'.join(texts))
        print(f'number of lines: {len(texts)}')

def reduce_wikidump(wikicorpus, factor):
    os.system(f'mv data/wikidump/{wikicorpus}_text.txt data/wikidump/{wikicorpus}_text_large.txt')
    small = ''
    with open(f'data/wikidump/{wikicorpus}_text_large.txt') as f:
        for i, line in enumerate(f):
            if i%factor==1:
                small += line
    with open(f'data/wikidump/{wikicorpus}_text.txt', 'w') as f:
        f.write(small)

def create_corpus_batch(wikicorpus, begin=0, end=10):
    '''
    create a batche of wpe corpus,
    be careful, sentences separators
    '''
    with open(f'data/wikidump/{wikicorpus}_text.txt') as f:
        lines = f.read().split('\n')

    with open(f'corpus/{wikicorpus}/stopwords.txt') as f:
        stopwords = f.read().strip().split('\n')

    n_lines = len(lines) // 10 # Number of line per batch
    
    for i in range(begin, end):
        ls = lines[i * n_lines: (i + 1) * n_lines]
        text = ''    

        if wikicorpus == 'wiki_th':
            for l in tqdm(ls): 
                tokens = tokenize(l)
                tokens = [t for t in tokens if re.match('^[ก-์]+$', t)] # thai word only
                tokens = [t for t in tokens if t not in stopwords] # not stopword
                if len(tokens) > 1:
                    text += ' '.join(tokens) + '\n'

        elif wikicorpus == 'wiki_cn':
            for l in tqdm(ls): 
                sents = re.split('[。]', l)
                for sent in sents:
                    tokens = [t for t in sent.strip().split() if re.match('^[\u4e00-\u9fa5]+$', t)] # chinese char only
                    tokens = [t for t in tokens if t not in stopwords] # not stopword
                    if len(tokens) > 1:
                        text += ' '.join(tokens) + '\n'

        elif wikicorpus == 'wiki_en':
            for l in tqdm(ls): 
                sents = sent_tokenize(l)
                for sent in sents:
                    tokens = [t for t in word_tokenize(sent) if re.match('^[a-zA-Z]+$', t)] # english word only
                    tokens = [t for t in tokens if t not in stopwords]
                    if len(tokens) > 1:
                        text += ' '.join(tokens) + '\n'


        output_path = f'corpus/{wikicorpus}/corpus_{i}.txt'
        with open(output_path, 'w') as f:
            f.write(text)

def create_corpus_combind(dir_):
    '''
    combind corpus_i.txt into corpus.txt
    '''
    text = ''
    # read each file
    for i in range(10):
        with open(f'corpus/{dir_}/corpus_{i}.txt') as f:
            obj = f.read()
            text += obj + '\n'
    
    # write final file
    with open(f'corpus/{dir_}/corpus.txt', 'w') as f:
        f.write(text.strip()) # remove last '\n'

def create_top_bigrams_old(wikicorpus, n_bigrams):
    # get tokens from corpus
    with open(f'corpus/{wikicorpus}/corpus.txt') as f:
        corpus = f.read().split('\n')
        tokens = [token for line in corpus for token in line.split()]

    # find bigrams
    with time_('bigram'):
        bigrams = nltk.collocations.BigramAssocMeasures()
        bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(tokens)

    for type_ in ['freq', 't', 'chi']:
        with time_(type_):
            if type_ == 'chi':
                bigramTable = bigramFinder.score_ngrams(bigrams.chi_sq)
            elif type_ == 't':
                bigramTable = bigramFinder.score_ngrams(bigrams.student_t)
            elif type_ == 'freq':
                bigramTable = bigramFinder.score_ngrams(bigrams.raw_freq)

            texts = ['\t'.join(bigram) for bigram, _ in bigramTable[:n_bigrams]]
            with open(f'corpus/{wikicorpus}/top_bigrams_{type_}.txt', 'w') as f:
                f.write('\n'.join(texts))

def create_top_bigrams(corpus, n_bigrams):
    # get tokens from corpus
    with open(f'corpus/{corpus}/corpus_word.txt') as f:
        corpus_ = f.read().split('\n')
        corpus_ = [l for l in corpus_ if l != '']
        tokens = [token for line in corpus_ for token in line.split()]

    # find bigrams
    with time_('bigram'):
        bigrams = nltk.collocations.BigramAssocMeasures()
        bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(tokens)

    for type_ in ['freq', 't', 'chi']:
        with time_(type_):
            if type_ == 'chi':
                bigramTable = bigramFinder.score_ngrams(bigrams.chi_sq)
            elif type_ == 't':
                bigramTable = bigramFinder.score_ngrams(bigrams.student_t)
            elif type_ == 'freq':
                bigramTable = bigramFinder.score_ngrams(bigrams.raw_freq)

            texts = ['\t'.join(bigram) for bigram, _ in bigramTable[:n_bigrams]]
            with open(f'corpus/{corpus}/top_bigrams_{type_}.txt', 'w') as f:
                f.write('\n'.join(texts))


