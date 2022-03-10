from ruword2tags import RuWord2Tags


if __name__ == '__main__':
    word2tags = RuWord2Tags()
    word2tags.load()

    for word in u'кошки рой для'.split():
        for i, tagset in enumerate(word2tags[word]):
            print(u'{}[{}] => {}'.format(word, i, tagset))
