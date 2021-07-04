#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]

def load_word_id_mapping(word_id_file, encoding='utf8'):
    """
    :param word_id_file: word-id mapping file path
    :param encoding: file's encoding, for changing to unicode
    :return: word-id mapping, like hello=5
    """
    word_to_id = dict()
    for line in open(word_id_file):
        line = line.decode(encoding, 'ignore').lower().split()
        word_to_id[line[0]] = int(line[1])
    print('\nload word-id mapping done!\n')
    return word_to_id


def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        cnt += 1
        line = line.split()
        # line = line.split()
        if len(line) != embedding_dim + 1:
            print('a bad word embedding: {}'.format(line[0]))
            continue
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print(np.shape(w2v))
    word_dict['$t$'] = (cnt + 1)
    # w2v -= np.mean(w2v, axis=0)
    # w2v /= np.std(w2v, axis=0)
    print(word_dict['$t$'], len(w2v))
    return word_dict, w2v


def load_word_embedding(word_id_file, w2v_file, embedding_dim, is_skip=False):
    word_to_id = load_word_id_mapping(word_id_file)
    word_dict, w2v = load_w2v(w2v_file, embedding_dim, is_skip)
    cnt = len(w2v)
    for k in word_to_id.keys():
        if k not in word_dict:
            word_dict[k] = cnt
            w2v = np.row_stack((w2v, np.random.uniform(-0.01, 0.01, (embedding_dim,))))
            cnt += 1
    print(len(word_dict), len(w2v))
    return word_dict, w2v


def load_aspect2id(input_file, word_id_mapping, w2v, embedding_dim):
    aspect2id = dict()
    a2v = list()
    a2v.append([0.] * embedding_dim)
    cnt = 0
    for line in open(input_file):
        line = line.lower().split()
        cnt += 1
        aspect2id[' '.join(line[:-1])] = cnt
        tmp = []
        for word in line:
            if word in word_id_mapping:
                tmp.append(w2v[word_id_mapping[word]])
        if tmp:
            a2v.append(np.sum(tmp, axis=0) / len(tmp))
        else:
            a2v.append(np.random.uniform(-0.01, 0.01, (embedding_dim,)))
    print(len(aspect2id), len(a2v))
    return aspect2id, np.asarray(a2v, dtype=np.float32)


def change_y_to_onehot(y):
    from collections import Counter
    print(Counter(y))
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    print(y_onehot_mapping)
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_inputs_twitter(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')

    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    target_words = []
    tar_len = []
    all_target, all_sent, all_y = [], [], []
    # read in txt file
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        # targets
        words = lines[i + 1].lower().split()
        target = words

        target_word = []
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        # sentiment
        y.append(lines[i + 2].strip().split()[0])

        # left and right context
        words = lines[i].lower().split()
        sent = words
        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
        if type_ == 'TD' or type_ == 'TC':
            # words_l.extend(target_word)
            words_l = words_l[:sentence_len]
            words_r = words_r[:sentence_len]
            sen_len.append(len(words_l))
            x.append(words_l + [0] * (sentence_len - len(words_l)))
            # tmp = target_word + words_r
            tmp = words_r
            if is_r:
                tmp.reverse()
            sen_len_r.append(len(tmp))
            x_r.append(tmp + [0] * (sentence_len - len(tmp)))
            all_sent.append(sent)
            all_target.append(target)
        else:
            words = words_l + target_word + words_r
            words = words[:sentence_len]
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))
    all_y = y;
    y = change_y_to_onehot(y)
    if type_ == 'TD':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
               np.asarray(sen_len_r), np.asarray(y)
    elif type_ == 'TC':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), np.asarray(sen_len_r), \
               np.asarray(y), np.asarray(target_words), np.asarray(tar_len), np.asarray(all_sent), np.asarray(all_target), np.asarray(all_y)
    elif type_ == 'IAN':
        return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
               np.asarray(tar_len), np.asarray(y)
    else:
        return np.asarray(x), np.asarray(sen_len), np.asarray(y)


def load_inputs_twitter_(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')

    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    target_words = []
    tar_len = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        words = lines[i + 1].decode(encoding).lower().split()
        # target_word = map(lambda w: word_to_id.get(w, 0), target_word)
        # target_words.append([target_word[0]])

        target_word = []
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        y.append(lines[i + 2].strip().split()[0])

        words = lines[i].decode(encoding).lower().split()
        words_l, words_r = [], []
        flag = 0
        puncs = [',', '.', '!', ';', '-', '(']
        for word in words:
            if word == '$t$':
                flag = 1
            if flag == 1 and word in puncs:
                flag = 2
            if flag == 2:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
            else:
                if word == '$t$':
                    words_l.extend(target_word)
                else:
                    if word in word_to_id:
                        words_l.append(word_to_id[word])
        if type_ == 'TD' or type_ == 'TC':
            words_l = words_l[:sentence_len]
            sen_len.append(len(words_l))
            x.append(words_l + [0] * (sentence_len - len(words_l)))
            tmp = words_r[:sentence_len]
            if is_r:
                tmp.reverse()
            sen_len_r.append(len(tmp))
            x_r.append(tmp + [0] * (sentence_len - len(tmp)))
        else:
            words = words_l + target_word + words_r
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))

    y = change_y_to_onehot(y)
    print(x)
    print(x_r)
    if type_ == 'TD':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
               np.asarray(sen_len_r), np.asarray(y)
    elif type_ == 'TC':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), np.asarray(sen_len_r), \
               np.asarray(y), np.asarray(target_words), np.asarray(tar_len)
    else:
        return np.asarray(x), np.asarray(sen_len), np.asarray(y)

def extract_aspect_to_id(input_file, aspect2id_file):
    dest_fp = open(aspect2id_file, 'w')
    lines = open(input_file).readlines()
    targets = set()
    for i in range(0, len(lines), 3):
        target = lines[i + 1].lower().split()
        targets.add(' '.join(target))
    aspect2id = list(zip(targets, range(1, len(lines) + 1)))
    for k, v in aspect2id:
        dest_fp.write(k + ' ' + str(v) + '\n')


def load_inputs_twitter_at(input_file, word_id_file, aspect_id_file, sentence_len, type_='', encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')
    if type(aspect_id_file) is str:
        aspect_to_id = load_aspect2id(aspect_id_file)
    else:
        aspect_to_id = aspect_id_file
    print('load aspect-to-id done!')

    x, y, sen_len = [], [], []
    aspect_words = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        aspect_word = ' '.join(lines[i + 1].lower().split())
        aspect_words.append(aspect_to_id.get(aspect_word, 0))

        y.append(lines[i + 2].split()[0])

        words = lines[i].decode(encoding).lower().split()
        ids = []
        for word in words:
            if word in word_to_id:
                ids.append(word_to_id[word])
        # ids = list(map(lambda word: word_to_id.get(word, 0), words))
        sen_len.append(len(ids))
        x.append(ids + [0] * (sentence_len - len(ids)))
    cnt = 0
    for item in aspect_words:
        if item > 0:
            cnt += 1
    print('cnt=', cnt)
    y = change_y_to_onehot(y)
    for item in x:
        if len(item) != sentence_len:
            print('aaaaa=', len(item))
    x = np.asarray(x, dtype=np.int32)

    return x, np.asarray(sen_len), np.asarray(aspect_words), np.asarray(y)


def load_inputs_sentence(input_file, word_id_file, sentence_len, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')

    x, y, sen_len = [], [], []
    for line in open(input_file):
        line = line.lower().decode('utf8', 'ignore').split('||')
        y.append(line[0])

        words = ' '.join(line[1:]).split()
        xx = []
        i = 0
        for word in words:
            if word in word_to_id:
                xx.append(word_to_id[word])
                i += 1
                if i >= sentence_len:
                    break
        sen_len.append(len(xx))
        xx = xx + [0] * (sentence_len - len(xx))
        x.append(xx)
    y = change_y_to_onehot(y)
    print('load input {} done!'.format(input_file))

    return np.asarray(x), np.asarray(sen_len), np.asarray(y)


def load_inputs_document(input_file, word_id_file, max_sen_len, max_doc_len, _type=None, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')

    x, y, sen_len, doc_len = [], [], [], []
    for line in open(input_file):
        line = line.lower().decode('utf8', 'ignore').split('||')
        # y.append(line[0])

        t_sen_len = [0] * max_doc_len
        t_x = np.zeros((max_doc_len, max_sen_len))
        doc = ' '.join(line[1:])
        sentences = doc.split('<sssss>')
        i = 0
        pre = ''
        flag = False
        for sentence in sentences:
            j = 0
            if _type == 'CNN':
                sentence = pre + ' ' + sentence
                if len(sentence.split()) < 5:
                    pre = sentence
                    continue
                else:
                    pre = ''
            for word in sentence.split():
                if j < max_sen_len:
                    if word in word_to_id:
                        t_x[i, j] = word_to_id[word]
                        j += 1
                else:
                    break
            t_sen_len[i] = j
            i += 1
            flag = True
            if i >= max_doc_len:
                break
        if flag:
            doc_len.append(i)
            sen_len.append(t_sen_len)
            x.append(t_x)
            y.append(line[0])

    y = change_y_to_onehot(y)
    print('load input {} done!'.format(input_file))

    return np.asarray(x), np.asarray(y), np.asarray(sen_len), np.asarray(doc_len)


def load_inputs_document_nohn(input_file, word_id_file, max_sen_len, _type=None, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')

    x, y, sen_len = [], [], []
    for line in open(input_file):
        line = line.lower().decode('utf8', 'ignore').split('||')
        words = ' '.join(line[1:]).split()
        i = 0
        tx = []
        for word in words:
            if i < max_sen_len:
                if word in word_to_id:
                    tx.append(word_to_id[word])
                    i += 1
        sen_len.append(i)
        x.append(tx + [0] * (max_sen_len - i))
        y.append(line[0])

    y = change_y_to_onehot(y)
    print('load input {} done!'.format(input_file))

    return np.asarray(x), np.asarray(y), np.asarray(sen_len)


def load_sentence(src_file, word2id, max_sen_len, freq=5):
    sf = open(src_file)
    x1, x2, len1, len2, y = [], [], [], [], []
    def get_q_id(q):
        i = 0
        tx = []
        for word in q:
            if i < max_sen_len and word in word2id:
                tx.append(word2id[word])
                i += 1
        tx += ([0] * (max_sen_len - i))
        return tx, i
    for line in sf:
        line = line.lower().split(' || ')
        q1 = line[0].split()
        q2 = line[1].split()
        is_d = line[2][0]
        tx, l = get_q_id(q1)
        x1.append(tx)
        len1.append(l)
        tx, l = get_q_id(q2)
        x2.append(tx)
        len2.append(l)
        y.append(is_d)
    index = range(len(y))
    # np.random.shuffle(index)
    x1 = np.asarray(x1, dtype=np.int32)
    x2 = np.asarray(x2, dtype=np.int32)
    len1 = np.asarray(len1, dtype=np.int32)
    len2 = np.asarray(len2, dtype=np.int32)
    y = change_y_to_onehot(y)
    return x1, x2, len1, len2, y

def load_inputs_cabasc(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')

    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    sent_short_final, sent_final = [], []
    target_words = []
    tar_len = []
    mult_mask = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        words = lines[i + 1].lower().split()
        # target_word = map(lambda w: word_to_id.get(w, 0), target_word)
        # target_words.append([target_word[0]])

        target_word = []
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        y.append(lines[i + 2].strip().split()[0])

        words = lines[i].lower().split()
        words_l, words_r, sent_short, sent= [], [], [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
        if type_ == 'TD' or type_ == 'TC':

            mult = [1]*sentence_len
            mult[len(words_l):len(words_l)+l] = [0.5] * l
            mult_mask.append(mult)

            sent_short.extend(words_l + target_word + words_r)
            words_l.extend(target_word)
            words_l = words_l[:sentence_len]
            words_r[:0] = target_word
            words_r = words_r[:sentence_len]
            sen_len_r.append(len(words_r))
            x_r.append([0] * (sentence_len - len(words_r)) + words_r)
            # tmp = target_word + words_r
            tmp = words_l
            if is_r:
                tmp.reverse()
            sen_len.append(len(tmp))
            x.append([0] * (sentence_len - len(tmp)) + tmp)
            sent_short_final.append(sent_short)
            sent_final.append(sent_short + [0] * (sentence_len - len(sent_short)))
        else:
            words = words_l + target_word + words_r
            words = words[:sentence_len]
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))
        if i == 0 :
            print('words left:{} \n length left: {} \n words right: {}\n length left: {}\n target: {}\n target length:{} \n sentiment: {}\n sentence:{}\n mask:{}'.format(
                x,
                sen_len,
                x_r,
                sen_len_r,
                target_words,
                tar_len,
                y,
                sent_final,
                mult_mask
            ))

    y = change_y_to_onehot(y)
    if type_ == 'TD':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
               np.asarray(sen_len_r), np.asarray(y)
    elif type_ == 'TC':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), np.asarray(sen_len_r), \
               np.asarray(y), np.asarray(target_words), np.asarray(tar_len), np.asarray(sent_short_final), np.asarray(sent_final), np.asarray(mult_mask)
    elif type_ == 'IAN':
        return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
               np.asarray(tar_len), np.asarray(y)
    else:
        return np.asarray(x), np.asarray(sen_len), np.asarray(y)

def load_inputs_full(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')

    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    sent_final = []
    target_words = []
    tar_len = []
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        words = lines[i + 1].lower().split()
        # target_word = map(lambda w: word_to_id.get(w, 0), target_word)
        # target_words.append([target_word[0]])

        target_word = []
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        y.append(lines[i + 2].strip().split()[0])

        words = lines[i].lower().split()
        words_l, words_r, sent = [], [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
        if type_ == 'TD' or type_ == 'TC':
            # words_l.extend(target_word)
            words_l = words_l[:sentence_len]
            words_r = words_r[:sentence_len]
            sent.extend(words_l + target_word + words_r)
            sen_len.append(len(words_l))
            x.append(words_l + [0] * (sentence_len - len(words_l)))
            # tmp = target_word + words_r
            tmp = words_r
            if is_r:
                tmp.reverse()
            sen_len_r.append(len(tmp))
            x_r.append(tmp + [0] * (sentence_len - len(tmp)))
            sent_final.append(sent+ [0] * (sentence_len - len(sent)))
        else:
            words = words_l + target_word + words_r
            words = words[:sentence_len]
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))
            

    y = change_y_to_onehot(y)
    if type_ == 'TD':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
               np.asarray(sen_len_r), np.asarray(y)
    elif type_ == 'TC':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), np.asarray(sen_len_r), \
               np.asarray(y), np.asarray(target_words), np.asarray(tar_len), np.asarray(sent_final)
    elif type_ == 'IAN':
        return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
               np.asarray(tar_len), np.asarray(y)
    else:
        return np.asarray(x), np.asarray(sen_len), np.asarray(y)


def calculate_classifications(ty_train, py_train, prob_train):
    # Convert to the correct format
    ty_train = np.asarray(ty_train)
    py_train = np.asarray(py_train)
    prob_train = np.asarray(prob_train)
    cc_p1, cc_p2, cc_p3, cc_f12, cc_f13, cc_f21, cc_f23, cc_f31, cc_f32 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    pcc_p1, pcc_p2, pcc_p3, pcc_f12, pcc_f13, pcc_f21, pcc_f23, pcc_f31, pcc_f32 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    # increase the counts when needed
    for index in range(len(ty_train)):
        if ty_train[index] == 0 and py_train[index] == 0:
            cc_p1 += 1
        if ty_train[index] == 0 and py_train[index] == 1:
            cc_f21 += 1
        if ty_train[index] == 0 and py_train[index] == 2:
            cc_f31 += 1
        if ty_train[index] == 1 and py_train[index] == 0:
            cc_f12 += 1
        if ty_train[index] == 1 and py_train[index] == 1:
            cc_p2 += 1
        if ty_train[index] == 1 and py_train[index] == 2:
            cc_f32 += 1
        if ty_train[index] == 2 and py_train[index] == 0:
            cc_f13 += 1
        if ty_train[index] == 2 and py_train[index] == 1:
            cc_f23 += 1
        if ty_train[index] == 2 and py_train[index] == 2:
            cc_p3 += 1
    # also for probabilistic
    for index in range(len(ty_train)):
        if ty_train[index] == 0:
            pcc_p1 += prob_train[index][0]
            pcc_f21 += prob_train[index][1]
            pcc_f31 += prob_train[index][2]
        if ty_train[index] == 1:
            pcc_f12 += prob_train[index][0]
            pcc_p2 += prob_train[index][1]
            pcc_f32 += prob_train[index][2]
        if ty_train[index] == 2:
            pcc_f13 += prob_train[index][0]
            pcc_f23 += prob_train[index][1]
            pcc_p3 += prob_train[index][2]

    # calculate the true/false preediction rates
    CC_T1 = (cc_p1) / (cc_p1 + cc_f21 + cc_f31 + 0.0001)
    CC_T2 = (cc_p2) / (cc_p2 + cc_f12 + cc_f32 + 0.0001)
    CC_T3 = (cc_p3) / (cc_p3 + cc_f23 + cc_f23 + 0.0001)
    CC_F12 = (cc_f12) / (cc_f12 + cc_p2 + cc_f32 + 0.0001)
    CC_F13 = (cc_f13) / (cc_f13 + cc_f23 + cc_p3 + 0.0001)
    CC_F21 = (cc_f21) / (cc_p1 + cc_f21 + cc_f31 + 0.0001)
    CC_F23 = (cc_f23) / (cc_f13 + cc_f23 + cc_p3 + 0.0001)
    CC_F31 = (cc_f31) / (cc_p1 + cc_f21 + cc_f31 + 0.0001)
    CC_F32 = (cc_f32) / (cc_f12 + cc_p2 + cc_f32 + 0.0001)

    PCC_T1 = (pcc_p1) / (pcc_p1 + pcc_f21 + pcc_f31 + 0.0001)
    PCC_T2 = (pcc_p2) / (pcc_p2 + pcc_f12 + pcc_f32 + 0.0001)
    PCC_T3 = (pcc_p3) / (pcc_p3 + pcc_f23 + pcc_f23 + 0.0001)
    PCC_F12 = (pcc_f12) / (pcc_f12 + pcc_p2 + pcc_f32 + 0.0001)
    PCC_F13 = (pcc_f13) / (pcc_f13 + pcc_f23 + pcc_p3 + 0.0001)
    PCC_F21 = (pcc_f21) / (pcc_p1 + pcc_f21 + pcc_f31 + 0.0001)
    PCC_F23 = (pcc_f23) / (pcc_f13 + pcc_f23 + pcc_p3 + 0.0001)
    PCC_F31 = (pcc_f31) / (pcc_p1 + pcc_f21 + pcc_f31 + 0.0001)
    PCC_F32 = (pcc_f32) / (pcc_f12 + pcc_p2 + pcc_f32 + 0.0001)

    # GENERAL
    # CC_T1 = 0.815950753384304
    # CC_T2 = 0.9378316195730387
    # CC_T3 = 0.1999997777780247
    # CC_F12 = 0.056861254218252145
    # CC_F13 = 0.24999965277826003
    # CC_F21 = 0.17177910597564297
    # CC_F23 = 0.49999930555652006
    # CC_F31 = 0.012269936141117354
    # CC_F32 = 0.005307050393703534
    #
    # PCC_T1 = 0.7556937055407879
    # PCC_T2 = 0.8932328235714556
    # PCC_T3 = 0.21932709209126056
    # PCC_F12 = 0.0814296399178251
    # PCC_F13 = 0.2973829439026427
    # PCC_F21 = 0.20195486429974477
    # PCC_F23 = 0.4498487034369941
    # PCC_F31 = 0.042351225660531684
    # PCC_F32 = 0.025337460695713448

    CC_count1, CC_count2, CC_count3 = 0, 0, 0
    # CC
    for i in range(len(prob_train)):
        if (np.argmax(prob_train[i])) == 0:
            CC_count1 += 1
        if (np.argmax(prob_train[i])) == 1:
            CC_count2 += 1
        if (np.argmax(prob_train[i])) == 2:
            CC_count3 += 1
    PCC_count1, PCC_count2, PCC_count3 = 0, 0, 0
    # PCC
    for i in range(len(prob_train)):
        PCC_count1 += prob_train[i][0]
        PCC_count2 += prob_train[i][1]
        PCC_count3 += prob_train[i][2]

    # calculate quantifications for each of the methods
    CC_share1 = CC_count1 / (CC_count1 + CC_count2 + CC_count3)
    CC_share2 = CC_count2 / (CC_count1 + CC_count2 + CC_count3)
    CC_share3 = CC_count3 / (CC_count1 + CC_count2 + CC_count3)

    PCC_share1 = PCC_count1 / (PCC_count1 + PCC_count2 + PCC_count3)
    PCC_share2 = PCC_count2 / (PCC_count1 + PCC_count2 + PCC_count3)
    PCC_share3 = PCC_count3 / (PCC_count1 + PCC_count2 + PCC_count3)

    ACC_share1 = (CC_share1 - ((CC_F12 - CC_F13) * (CC_share2 - CC_F23)) / (CC_T2 - CC_F23) - CC_F13) / \
                 (CC_T1 - ((CC_F12 - CC_F13) * (CC_F21 - CC_F23)) / (CC_T2 - CC_F23) - CC_F13)
    ACC_share2 = (CC_share2 - ((CC_F21 - CC_F23) * (CC_share1 - CC_F13)) / (CC_T1 - CC_F13) - CC_F23) / \
                 (CC_T2 - ((CC_F21 - CC_F23) * (CC_F12 - CC_F13)) / (CC_T1 - CC_F13) - CC_F23)
    ACC_share3 = 1 - ACC_share1 - ACC_share2

    PACC_share1 = (PCC_share1 - ((PCC_F12 - PCC_F13) * (PCC_share2 - PCC_F23)) / (PCC_T2 - PCC_F23) - PCC_F13) / \
                  (PCC_T1 - ((PCC_F12 - PCC_F13) * (PCC_F21 - PCC_F23)) / (PCC_T2 - PCC_F23) - PCC_F13)
    PACC_share2 = (PCC_share2 - ((PCC_F21 - PCC_F23) * (PCC_share1 - PCC_F13)) / (PCC_T1 - PCC_F13) - PCC_F23) / \
                  (PCC_T2 - ((PCC_F21 - PCC_F23) * (PCC_F12 - PCC_F13)) / (PCC_T1 - PCC_F13) - PCC_F23)
    PACC_share3 = 1 - PACC_share1 - PACC_share2
    return [CC_T1, CC_T2, CC_T3, CC_F12, CC_F13, CC_F21, CC_F23, CC_F31, CC_F31, CC_share1, CC_share2, CC_share3, PCC_share1, PCC_share2, PCC_share3, ACC_share1, ACC_share2, ACC_share3, PACC_share1, PACC_share2, PACC_share3]
