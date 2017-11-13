import cPickle
import pdb
import numpy as np
def idx2sent(text, alphabet):
    char_seq = []
    # print(text)
    for it in text:
        it_list = list(it)
        # padded = pad_sentence(it_list)
        # text_int8_repr = string_to_int8_conversion(padded, alphabet)
        text_int8_repr = string_to_int8_conversion(it_list, alphabet)
        char_seq.append(text_int8_repr)
    return char_seq


def pad_sentence(char_seq, padding_char=" ", char_seq_length=301):
    # char_seq_length = 1014

    num_padding = char_seq_length - len(char_seq)

    new_char_seq = char_seq + [padding_char] * num_padding
    return new_char_seq


def string_to_int8_conversion(char_seq, alphabet):
    x = [alphabet.find(char) + 2 for char in char_seq]
    # x = np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)
    return x

def prepare_data_for_charCNN(loadpath = "./data/yahoo4char.p"):

    x = cPickle.load(open(loadpath,"rb"))

    train, val, test                    = x[0], x[1], x[2]
    train_text, val_text, test_text     = x[3], x[4], x[5]
    train_lab, val_lab, test_lab        = x[6], x[7], x[8]
    wordtoix, ixtoword                  = x[9], x[10]

    # alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"

    alphabet = "abcdefghijklmnopqrstuvwxyz0,.!?()"



    train_char = idx2sent(train_text, alphabet)
    val_char = idx2sent(val_text, alphabet)
    test_char = idx2sent(test_text, alphabet)
    chartoix = { c: i + 2 for i, c in enumerate(alphabet)} # make sure 0 is space
    chartoix[' '] = 1
    ixtochar = { i+2:c for i, c in enumerate(alphabet)}
    ixtochar[1] = ' '


    # add padding character
    chartoix['N'] = 0
    ixtochar[0] = 'N'

    with open('./data/yahoo_char.p', 'w+') as f:
        cPickle.dump([train_char, val_char, test_char, train_text, val_text, test_text, train_lab, val_lab, test_lab, chartoix, ixtochar, alphabet, ], f)

if __name__ == '__main__':

    prepare_data_for_charCNN()
