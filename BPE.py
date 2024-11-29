import argparse
import re
from collections import Counter


def learn_bpe_algo(input_text, vocab_size):
    vocab = preprocess_text(input_text)
    merges = []
    pairs = get_frequent_pairs(vocab)
    for _ in range(vocab_size):
        best_pair = find_most_frequent_pair(pairs)
        if best_pair is None:
            break
        merges.append(best_pair)
        vocab, pairs = update_vocab_with_pair(best_pair, vocab, pairs)
    splits = {word.replace(" ", ""): "".join(word) for word in vocab.keys()}
    return splits, merges


def preprocess_text(text):
    vocab = Counter()
    for line in text.split('\n'):
        tokens = [list(word) + ['_'] for word in re.findall(r'\S+', line)]
        for token in tokens:
            vocab[" ".join(token)] += 1
    return vocab

def preprocess_text_apply(text):
    preprocessed = []
    words = re.findall(r'\S+|\s+', text)
    for word in words:
        if word.isspace():
            preprocessed.append(word)
            continue
        preprocessed.append(word + "_")
    return preprocessed

def preprocess_text_input(text):
    preprocessed = []
    words = re.findall(r'\S+|\s+', text)
    for word in words:
        if word.isspace():
            preprocessed.append(word)
            continue
        curr_words = list(word)
        curr_words.append("_")
        preprocessed.extend([curr_words])
    return preprocessed

def get_frequent_pairs(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def update_vocab_with_pair(pair, vocab, pairs):
    joined_pair = ''.join(pair)
    pair_str = ' '.join(pair)
    regex_pattern = re.compile(r'(?<!\S)' + re.escape(pair_str) + r'(?!\S)')
    updated_vocab = Counter()
    for word, freq in vocab.items():
        if pair_str in word:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    if i > 0:
                        pairs[symbols[i - 1], symbols[i]] -= freq
                        pairs[symbols[i - 1], joined_pair] += freq
                    if i + 2 < len(symbols):
                        pairs[symbols[i + 1], symbols[i + 2]] -= freq
                        pairs[joined_pair, symbols[i + 2]] += freq
                        symbols[i + 1] = joined_pair
                        i += 1
            modified_word = regex_pattern.sub(joined_pair, word)
            updated_vocab[modified_word] = vocab[word]
        else:
            updated_vocab[word] = freq
    pairs.pop(pair)
    return updated_vocab, pairs

def find_most_frequent_pair(freq_dict):
    max_freq = 0
    max_pair = None
    for pair, freq in freq_dict.items():
        if freq > max_freq:
            max_freq = freq
            max_pair = pair
        elif freq == max_freq and max_pair is not None:
            if pair[1] < max_pair[1]:
                max_pair = pair
            elif pair[1] == max_pair[1]:
                if pair[0] < max_pair[0]:
                    max_pair = pair
    return max_pair


def apply_bpe_algo(text, merges, outpath):
    word_map = {word + "_": list(word) + ["_"] for word in text.split()}
    for token in merges:
        for word, new_word in word_map.items():
            if "".join(token) in word:
                word_map[word] = merge_subwords(new_word, token)
    input = preprocess_text_apply(text)
    output = []
    for word in input:
        if word.isspace():
            output.append(word)
            continue
        output.append(" ".join(word_map[word]).replace("_", "").rstrip())
    with open(outpath, 'w', encoding='utf-8') as f:         
        f.write("".join(output))

def merge_subwords(word, new_token):
    new_word = []
    i = 0
    while i < len(word):
        if word[i] == new_token[0]:
            isMatch = True
            for j in range(1, len(new_token)):
                if i + j >= len(word) or word[i + j] != new_token[j]:
                    isMatch = False
            if isMatch:
                new_word.append("".join(new_token))
                i += len(new_token)
            else:
                new_word.append(word[i])
                i += 1
        else:
            new_word.append(word[i])
            i += 1
    return new_word
        
def main(args):
    inpath = args.inpath
    outpath = args.outpath
    vocab = args.vocab
    vocab_size = args.vocab_size
    learn_bpe = args.learn_bpe
    apply_bpe = args.apply_bpe
    file = open(inpath, 'r',encoding='utf-8')
    content = file.read()
    if learn_bpe:
        splits, merges = learn_bpe_algo(content, vocab_size)
        with open(vocab, 'w', encoding='utf-8') as f:
            for i in range(len(merges)):
                if i == len(merges) - 1:
                    f.write(merges[i][0] + " " + merges[i][1])
                else:
                    f.write(merges[i][0] + " " + merges[i][1] + '\n')
        input = preprocess_text_apply(content)
        output = []
        for word in input:
            if word.isspace():
                output.append(word)
                continue
            output.append(splits[word].replace("_", "").rstrip())
        with open(outpath, 'w', encoding='utf-8') as f:
            f.write(''.join(output))
    elif apply_bpe:
        with open(vocab, 'r', encoding='utf-8') as f:
            merges = [tuple(line.strip("\n").split(" ")) for line in f]
        apply_bpe_algo(content, merges, outpath)
    else:
        print('Please specify if you want to learn or apply BPE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process command-line arguments for BPE learning.")
    parser.add_argument('--learn_bpe', action='store_true', help='Flag to trigger learning BPE')
    parser.add_argument('--apply_bpe', action='store_true', help='Flag to trigger applying BPE')
    parser.add_argument('--inpath', type=str, required=True, help='Path to input text')
    parser.add_argument('--outpath', type=str, required=True, help='Path to output text')
    parser.add_argument('--vocab', type=str, required=True, help='Path to vocab file')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Size of the vocabulary')
    args = parser.parse_args()
    main(args)