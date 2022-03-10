# -*- coding: utf-8 -*-
"""
Вариант грамматического словаря для генеративных языковых моделей.
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import io
import gzip
import pickle
import os
import pathlib


class RuFlexer:
    dict_filename = 'ruflexer.dat'

    def __init__(self):
        pass

    def split_tag(self, tag):
        return tuple(tag.split(':'))

    def split_tags(self, tags_str):
        return [self.split_tag(tag) for tag in tags_str.split(' ')]

    def is_good(self, tags_str):
        # Исключаем краткие формы прилагательных в среднем роде, так как
        # они обычно омонимичны с более употребимыми наречиями.
        return u'КРАТКИЙ:1 ПАДЕЖ:ИМ РОД:СР' not in tags_str

    @staticmethod
    def decode_pos(pos):
        if pos in [u'ДЕЕПРИЧАСТИЕ', u'ГЛАГОЛ', u'ИНФИНИТИВ']:
            return u'ГЛАГОЛ'
        else:
            return pos

    def build(self, path):
        self.prepositions = set()
        self.lemma2forms = dict()
        self.word2pos = dict()
        self.word2tags = dict()
        self.tagstr2id = dict()
        self.tagsid2list = dict()

        self.word2pos = dict()

        with io.open(path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('\t')
                if len(tx) == 5:
                    word = tx[0].replace(u' - ', u'-')
                    tags_str = tx[3]
                    if self.is_good(tags_str):
                        tags_str = tags_str.replace(u'ПЕРЕЧИСЛИМОСТЬ:ДА', u'')\
                            .replace(u'ПЕРЕЧИСЛИМОСТЬ:НЕТ', u'')\
                            .replace(u'ПЕРЕХОДНОСТЬ:ПЕРЕХОДНЫЙ', u'')\
                            .replace(u'ПЕРЕХОДНОСТЬ:НЕПЕРЕХОДНЫЙ', u'')

                        pos0 = tx[1]

                        # НАЧАЛО ОТЛАДКИ
                        #if word == u'профессиональный':
                        #    print('')
                        # КОНЕЦ ОТЛАДКИ

                        lemma = tx[2].replace(u' - ', u'-')
                        pos = RuFlexer.decode_pos(pos0)
                        self.add_word(word, lemma, pos, tags_str)

        output_file = '../tmp/'+self.dict_filename
        with gzip.open(output_file, 'w') as f:
            pickle.dump((self.prepositions, self.lemma2forms, self.word2pos, self.word2tags, self.tagstr2id, self.tagsid2list), f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Сохранен файл словаря размером {:d} Мб'.format(int(os.path.getsize(output_file)/1000000)))

    def add_word(self, word, lemma, pos, tags_str0):
        self.word2pos[word] = pos
        if pos == u'ПРЕДЛОГ':
            self.prepositions.add(word)

        tags_str = u'ЧАСТЬ_РЕЧИ:'+pos + u' ' + tags_str0

        # формы прилагательных в винительном падеже дополняем тегами ОДУШ:ОДУШ и ОДУШ:НЕОДУШ,
        # если не указан тег ОДУШ:НЕОДУШ
        if pos == u'ПРИЛАГАТЕЛЬНОЕ' and u'ПАДЕЖ:ВИН' in tags_str and u'ОДУШ:' not in tags_str:
            tags_str += u' ОДУШ:ОДУШ ОДУШ:НЕОДУШ'

        tags_str = tags_str.replace(u'  ', u' ')
        if tags_str not in self.tagstr2id:
            tags_id = len(self.tagstr2id)
            self.tagstr2id[tags_str] = tags_id
            self.tagsid2list[tags_id] = self.split_tags(tags_str)
        else:
            tags_id = self.tagstr2id[tags_str]

        if word not in self.word2tags:
            self.word2tags[word] = [tags_id]
        else:
            self.word2tags[word].append(tags_id)

        if lemma not in self.lemma2forms:
            self.lemma2forms[lemma] = [(word, tags_id)]
        else:
            self.lemma2forms[lemma].append((word, tags_id))

    def load(self):
        module_folder = str(pathlib.Path(__file__).resolve().parent)
        p = os.path.join(module_folder, '../output', self.dict_filename)
        if not os.path.exists(p):
            p = os.path.join(module_folder, self.dict_filename)

        with gzip.open(p, 'r') as f:
            self.prepositions, self.lemma2forms, self.word2pos, self.word2tags, self.tagstr2id, self.tagsid2list = pickle.load(f)

    def __contains__(self, word):
        return word in self.word2pos

    def get_forms(self):
        return self.word2pos.keys()

    def get_pos(self, word):
        if word in self.word2pos:
            return self.word2pos[word]
        else:
            return None

    def get_word_tagsets(self, word):
        tagsets = []
        for tagset_id in self.word2tags[word]:
            tagsets.append(self.tagsid2list[tagset_id])
        return tagsets

    def find_forms_by_tags(self, lemma, tags):
        if lemma in self.lemma2forms:
            for form, tagset_id in set(self.lemma2forms[lemma]):
                tagset = self.tagsid2list[tagset_id]
                if all((tag in tagset) for tag in tags):
                    yield form

    def all_prepositions(self):
        return self.prepositions


if __name__ == '__main__':
    flexer = RuFlexer()
    flexer.build('../data/word2tags.dat')
