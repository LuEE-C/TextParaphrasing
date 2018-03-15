import numpy as np
import os
from string import ascii_lowercase as al
from collections import Counter
import pickle

# This does two passes through the data, the first one to figure out what are the characters
# or words that interest us, getting rid of those that are not present enough
def convert_text_to_nptensor(directory='../datas/BillionWords/', cutoff=5, min_frequency_words=50000, max_lines=50000000, name='Billion_Words'):
    if os.path.isfile('../datas/TransformedData/' + 'text_' + name + str(cutoff) + '_' + str(min_frequency_words) + '.npy'):
        X = np.load('../datas/TransformedData/' +  'text_' + name + str(cutoff) + '_' + str(min_frequency_words) + '.npy')
        ind_to_word = pickle.load(open('../datas/TransformedData/' + 'ind_to_word_' + name + str(cutoff) + '_' + str(min_frequency_words) + '.pickle', 'rb'))
    else:
        words = Counter()
        n_lines = 0
        files = []

        # First pass to gather statistics on data
        for file in os.listdir(directory):
                files.append(directory + file)
        for file_idx in range(len(files)):
            with open(files[file_idx], encoding='utf-8') as f:
                text = f.readlines()
            for line in text:
                line = line.replace('\n', '').replace('.', '').replace('!', '').replace('?', '').replace('\t', ' ')
                line = line.lower()
                words.update(line.split(' '))
                n_lines += len(line.split(' '))//cutoff

        number_of_words, removed = 0, 0
        words_init = len(words)

        for k in list(words):
            number_of_words += words[k]
            if words[k] < min_frequency_words:
                removed += words[k]
                del words[k]
        print('% of raw words remaining :', (number_of_words - removed)/number_of_words*100.0)
        print('Initial amount of tokens :', words_init)
        print('Current amount of tokens :', len(words))
        print('% of remaining tokens :', len(words)/words_init)
        print('Max amount of lines :', n_lines)

        # We reserve 0 for 0 padding
        word_to_ind = dict((c, i) for i, c in enumerate(list(set(words))))
        ind_to_word = dict((i, c) for i, c in enumerate(list(set(words))))
        X = np.zeros((max_lines, cutoff), dtype=np.int16)

        lines_added = 0

        for file_idx in range(len(files)):
            with open(files[file_idx], encoding='utf-8') as f:
                text = f.readlines()
            for line in text:
                line = line.replace('\n', '').replace('.', '').replace('!', '').replace('?', '').replace('\t', ' ')
                line = line.lower()
                line = line.split(' ')
                offset = 0
                while(len(line) > offset + cutoff) & (lines_added < max_lines):

                    # This makes sure that every word in the coming section of text is in the vocabulary, else we pass it
                    check_word = True
                    for word in line[offset: offset + cutoff]:
                        try:
                            word_to_ind[word]
                        except KeyError:
                            check_word = False
                    if check_word == False:
                        offset += cutoff

                    else:
                        for t, word in enumerate(line[offset: offset + cutoff]):
                            X[lines_added, t] = word_to_ind[word]

                        offset += cutoff
                        lines_added += 1
        print(lines_added)
        with open('../datas/TransformedData/ind_to_word_' + name + str(cutoff) + '_' + str(min_frequency_words) + '.pickle', 'wb') as pck:
            pickle.dump(ind_to_word, pck)
        np.save('../datas/TransformedData/text_' + name + str(cutoff) + '_' + str(min_frequency_words), X[:lines_added])
    return ind_to_word, X


# This does two passes through the data, the first one to figure out what are the characters
# or words that interest us, getting rid of those that are not present enough
def convert_text_to_nptensor_word(directory='../datas/BillionWords/', cutoff=180, max_lines=50000000, name='Billion_Words'):
    if os.path.isfile('../datas/TransformedData/' + 'text_' + name + str(cutoff) + '_' + str(min_frequency_words) + '.npy'):
        X = np.load('../datas/TransformedData/' +  'text_' + name + str(cutoff) + '_' + str(min_frequency_words) + '.npy')
        ind_to_word = pickle.load(open('../datas/TransformedData/' + 'ind_to_word_' + name + str(cutoff) + '_' + str(min_frequency_words) + '.pickle', 'rb'))
    else:
        words = Counter()
        n_lines = 0
        files = []

        # First pass to gather statistics on data
        for file in os.listdir(directory):
                files.append(directory + file)
        for file_idx in range(len(files)):
            with open(files[file_idx], encoding='utf-8') as f:
                text = f.readlines()
            for line in text:
                line = line.replace('\n', '').replace('.', '').replace('!', '').replace('?', '').replace('\t', ' ')
                line = line.lower()
                words.update(line.split(' '))
                n_lines += len(line.split(' '))//cutoff

        number_of_words, removed = 0, 0
        words_init = len(words)

        for k in list(words):
            number_of_words += words[k]
            if words[k] < min_frequency_words:
                removed += words[k]
                del words[k]
        print('% of raw words remaining :', (number_of_words - removed)/number_of_words*100.0)
        print('Initial amount of tokens :', words_init)
        print('Current amount of tokens :', len(words))
        print('% of remaining tokens :', len(words)/words_init)
        print('Max amount of lines :', n_lines)

        # We reserve 0 for 0 padding
        word_to_ind = dict((c, i) for i, c in enumerate(list(set(words))))
        word_to_ind['_'] = len(list(set(words)))
        ind_to_word = dict((i, c) for i, c in enumerate(list(set(words))))
        ind_to_word[len(list(set(words)))] = '_'
        X = np.zeros((max_lines, cutoff), dtype=np.int16)

        lines_added = 0

        for file_idx in range(len(files)):
            with open(files[file_idx], encoding='utf-8') as f:
                text = f.readlines()
            for line in text:
                line = line.replace('\n', '').replace('.', '').replace('!', '').replace('?', '').replace('\t', ' ')
                line = line.lower()
                line = line.split(' ')
                offset = 0
                while(len(line) > offset + cutoff) & (lines_added < max_lines):

                    # This makes sure that every word in the coming section of text is in the vocabulary, else we pass it
                    check_word = True
                    for word in line[offset: offset + cutoff]:
                        try:
                            word_to_ind[word]
                        except KeyError:
                            check_word = False
                    if check_word == False:
                        offset += cutoff

                    else:
                        for t, word in enumerate(line[offset: offset + cutoff]):
                            X[lines_added, t] = word_to_ind[word]

                        offset += cutoff
                        lines_added += 1
        print(lines_added)
        with open('../datas/TransformedData/ind_to_word_' + name + str(cutoff) + '_' + str(min_frequency_words) + '.pickle', 'wb') as pck:
            pickle.dump(ind_to_word, pck)
        np.save('../datas/TransformedData/text_' + name + str(cutoff) + '_' + str(min_frequency_words), X[:lines_added])
    return ind_to_word, X



if __name__ == '__main__':
    convert_text_to_nptensor(cutoff=16, min_frequency_words=150000)