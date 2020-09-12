import re
import csv
import numpy as np
from collections import Counter
import random
import collections

def read_csv(path):
    """
    :param path: str
    :return: [[raw_data]], [col_names] =
            [
                [str,str,'',...],   - product
                [str,str,...],      - sub-product
                [str,str,...],      - issue
                ...                 - ...
             ],
             ['col_name1', 'col_name2', ...]
    """
    raw_data = []
    col_names = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for i, row in enumerate(reader):
            if i == 0:
                for col_name in row:
                    raw_data.append([])
                    col_names.append(col_name)
            else:
                for c_index, text in enumerate(row):
                    # if c_index == 5:
                    #     text = self.label_special_token(text)
                    raw_data[c_index].append(text)
    return col_names, raw_data

def write_state_cols(col_data, col_name, path):
    stat = Counter(col_data)
    full_path = path + '/' + col_name + ".csv"
    total = 0
    with open(full_path, 'w') as f:
        for key, value in stat.items():
            total = total + value
            f.write("%s,%d\n" % (key, value))
    return True

class Text_Processor:
    def __init__(self, rare_word_threshold):
        self.rare_word_threshold = rare_word_threshold

    def label_special_token(self, text):
        """
        :param text: str
        :return: labeled str
        NOTE: transform lowercase
        """
        text = re.sub("[X]+/[X]+/[X]+", ' <DATE> ', text)
        text = re.sub("[X]+/[X]+/[\d]*", ' <DATE> ', text)
        text = re.sub("(XXXX[ ]?)+", ' <NAME> ', text)
        text = re.sub("{\$\S+}", ' <DOLLAR> ', text)

        text = re.sub('..', ' <PERIOD> ', text)
        text = re.sub(',', ' <COMMA> ', text)
        text = re.sub('"', ' <QUOTATION_MARK> ', text)
        text = re.sub(';', ' <SEMICOLON> ', text)
        text = re.sub('!', ' <EXCLAMATION_MARK> ', text)
        text = re.sub('\?', ' <QUESTION_MARK> ', text)
        text = re.sub('[{｛﹛【《＜〈〔［\[（(]', ' <LEFT_PAREN> ', text)
        text = re.sub('[}｝﹜】》＞〉〕］\])）]', ' <RIGHT_PAREN> ', text)
        text = re.sub('-', ' <HYPHENS> ', text)
        text = re.sub('=', ' <EQUAL> ', text)
        text = re.sub('\?', ' <QUESTION_MARK> ', text)
        text = re.sub('/', ' <FORWARD_SLASH> ', text)
        text = re.sub(':', ' <COLON> ', text)

        text = text.lower()

        return text

    def split2vocab(self, col_data):
        """
        :param col_data: [str, str, str]
        :return: col_data: [[str], [str], [str], ...]
        """
        splited_col_data = []
        for row in col_data:
            splited_col_data.append(row.split())
        return splited_col_data

    def get_token_freq(self, col_data):
        """
        :param col_data: [[str], [str], [str], ...]
        :return: token_count = {"consumer": 423, "loan": 521, ...}
        """
        token_count = Counter()
        for row in col_data:
            token_count = token_count + Counter(row)
        return token_count

    def filter_out_word(self, splited_col_data, token_count):
        """
        :param splited_col_data: [[str], [str], [str], ...]
        :param token_count: {"consumer": 423, "loan": 521, ...}
        :return: filtered col_data: [[str], [str], [str], ...]
        NOTE update the token_count
        """
        for index, row in enumerate(splited_col_data):
            filtered_row = [token for token in row if token_count[token] > self.rare_word_threshold]
            splited_col_data[index] = filtered_row
        filtered_col_data = splited_col_data

        # update the token_count
        new_token_count = {}
        for word, count in token_count.items():
            if int(count) > self.rare_word_threshold:
                new_token_count[word] = int(count)
        return filtered_col_data, new_token_count

    def create_lookup_table(self, token_count):
        """
        :param token_count: {"consumer": 423, "loan": 521, ...}
        :return:
                    int2vocab = {1:'a',2:'b', ...}
                    vocab2int = {'a':1,'b':2, ...}
        """
        sorted_vocab = sorted(token_count, key=token_count.get, reverse=True)
        int2vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
        vocab2int = {word: ii for ii, word in int2vocab.items()}
        return int2vocab, vocab2int

    def vocab_into_int(self, col_data, vocab2int):
        """
        :param sentc: col_data: [[str], [str], [str], ...]
        :return: col_index_data: [[index], [index], [index], ...]
        """
        col_index_data = []
        for row in col_data:
            col_index_data.append([vocab2int[vocab] for vocab in row])
        return col_index_data

    def process(self, col_text_data):
        """
        :param path: str
        :made: int2vocab, vocab2int
        :made: index_col_data = [[index], [index], [index], ...]
        :return: namedtuple = Text_Processor_Output
        """

        Text_Processor_Output = collections.namedtuple("Text_Processor_Output", field_names=[
            "vocab2int", "int2word", "index_col_data"])

        # label the special token
        for i, row_text in enumerate(col_text_data):
            col_text_data[i] = self.label_special_token(row_text)

        # get token count
        splited_col_data = self.split2vocab(col_text_data)
        token_count = self.get_token_freq(splited_col_data)

        # filter out rare word
        filtered_col_data, token_count = self.filter_out_word(splited_col_data, token_count)

        # update word2int and int2word
        Text_Processor_Output.int2vocab, Text_Processor_Output.vocab2int = self.create_lookup_table(token_count)

        # transform word into index
        Text_Processor_Output.index_col_data = self.vocab_into_int(filtered_col_data, Text_Processor_Output.vocab2int)

        return Text_Processor_Output

class Batch_Generator:
    def __init__(self):
        pass

    def get_info_from_data(self, col_data):
        """
        :param col_data: ['a','b','d',...]
        :return: {0:'h', 1:'a', 2:'d', ...}, {'h':0, 'a':1, 'd':2, ...}
        """
        token_count = Counter(col_data)
        sorted_vocab = sorted(token_count, key=token_count.get, reverse=True)

        int2vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
        vocab2int = {word: ii for ii, word in int2vocab.items()}

        return token_count, int2vocab, vocab2int

    def label_data(self, col_data, word2int):
        """
        :param col_data: ['a','b','d',...]
        :use: self.domain_vocab2int / self.codomain_vocab2int
        :return: [0,1,2,3,...]
        """
        index_col_data = []
        for ele in col_data:
            index_col_data.append(word2int[ele])
        return index_col_data

    def systematize_category(self, domain_col, codomain_col):
        """
        :param domain_col:          [0,1,2,3,...]
        :param codomain_col:        [92,81,72,51,...]
        :return:                    {92: [1,3,2], 81: [3,2,4,5,7], 72: [1,2,5,7]...}
        """
        # transform to array
        domain_col = np.array(domain_col)
        codomain_col = np.array(codomain_col)

        # get keys from co-domain
        codomain_dict = Counter(codomain_col)
        codomain_keys = sorted(codomain_dict, key=codomain_dict.get, reverse=True)
        category = {}
        for codomain_key in codomain_keys:
            mask = domain_col[codomain_col==codomain_key]
            category[codomain_key] = list(domain_col[mask])
        return category

    def prepare_generator(self, domain_col, codomain_col):
        """
        :param domain_col:              ['cat_1', 'cat_3', 'cat_1', ...]
        :param codomain_col:            ['cat_B', 'cat_C', 'cat_B', ...]
        :made: domain_int2vocab, domain_vocab2int
        :made: codomain_int2vocab, codomain_vocab2int
        :made: category =          {codomain_1: [1,3,2], codomain_2: [3,2,4,5,7], codomain_3: [1,2,5,7]...}
        return namedtuple = Preparation

        """
        Preparation = collections.namedtuple("Preparation",field_names=[
            "domain_int2vocab", "domain_vocab2int", "codomain_int2vocab", "codomain_vocab2int", "category"])

        # get the domain_col token_count
        print("Get info from domain data ...")
        token_count, Preparation.domain_int2vocab, Preparation.domain_vocab2int = self.get_info_from_data(domain_col)
        # label the domain_col
        domain_index = self.label_data(domain_col, Preparation.domain_vocab2int)
        print("Successful.")

        # get the codomain_col token_count
        print("Get info from codomain data ...")
        token_count, Preparation.codomain_int2vocab, Preparation.codomain_vocab2int = self.get_info_from_data(codomain_col)
        # label the codomain_col
        codomain_index = self.label_data(codomain_col, Preparation.codomain_vocab2int)
        print("Successful.")

        # systematization {0: [1,3,1], 1: [4,2,1], ...}
        print("Systematize Category ...")
        Preparation.category = self.systematize_category(domain_index, codomain_index)
        print("Successful.")

        return Preparation

    def get_domain(self, codomain, category):
        """
        :param codomain: int
        :param: category = {'codomain_1': [1,3,2,...], 'codomain_2': [3,2,4,5,7,...], 'codomain_3': [1,2,5,7,...]...}
        :return: domains = [int]
        """
        return category[codomain]

    def create_fc_batches(self, category):
        """
        :param category: {'codomain_1':[1,2,3,4,...], 'codomain_2':[5,4,2,7,...], 'codomain_3':[9,1,6,5,...],...}
        :return:
                    x     = [1,0,2,1,1,2,1,...],                        (domain)
                    y     = [1,2,3,1,3,1,2,...],                        (codomain)
                    idx   = [98,58,12,18,51,...]
        """
        print("Creating whole batches...")
        fc_train_set = collections.namedtuple("fc_train_set", field_names=["train_x","train_y","shuffle_indexs"])

        # create batch_x, batch_y, batch_noise
        fc_train_set.train_x = []
        fc_train_set.train_y = []

        for codomain in category.keys():
            domains = self.get_domain(codomain, category)
            fc_train_set.train_x.extend(domains)
            fc_train_set.train_y.extend([codomain] * len(domains))

        # shuffle the training set
        fc_train_set.shuffle_indexs = list(range(len(fc_train_set.train_x)))
        random.shuffle(fc_train_set.shuffle_indexs)
        print("Successful.")

        return fc_train_set

    def get_fc_batches(self, batch_size, fc_train_set):
        """
        :param batch_size:                      int
        :param fc_train_set:
                                x     = [1,0,2,1,1,2,1,...],                        (domain)
                                y     = [1,2,3,1,3,1,2,...],                        (codomain)
                                idx   = [98,58,12,18,51,...]

        :made x:                [1,0,2,1,1,2,1,...]
        :made y:                [1,2,3,1,3,1,2,...]
        :yield: batch_x, batch_y
        """

        # find the last index which can completely divided by batch_size
        num_batches = len(fc_train_set.shuffle_indexs) // batch_size
        shuffle_indexs = fc_train_set.shuffle_indexs[:num_batches*batch_size]

        train_x, train_y = [], []
        # yield the batch_set
        for c, index in enumerate(shuffle_indexs):
            train_x.append(fc_train_set.train_x[index])
            train_y.append(fc_train_set.train_y[index])
            if (c+1) % batch_size == 0:
                yield train_x, train_y
                train_x, train_y = [], []







