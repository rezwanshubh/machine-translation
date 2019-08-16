import argparse
import json
import torch

def read_instances_from_file(inst_file, max_sent_len):
    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [['<BOS>'] + word_inst + ['<EOS>']]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts

def convert_instance_to_idx_seq(word_insts, word2idx):
    idx_seq_arr = []
    for instance in word_insts:
        idx_seq = []
        for token in instance:
            try:
                idx_seq.append(word2idx[token])
            except:
                idx_seq.append(word2idx['<UNK>'])
        idx_seq_arr.append(idx_seq)

    return idx_seq_arr



def main():
    print('test')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-train_data_src', required=True)
    arg_parser.add_argument('-train_data_tgt', required=True)
    arg_parser.add_argument('-valid_data_src', required=True)
    arg_parser.add_argument('-valid_data_tgt', required=True)
    arg_parser.add_argument('-vocab_data_src', required=True)
    arg_parser.add_argument('-vocab_data_tgt', required=True)
    arg_parser.add_argument('-save_all_data', required=True)
    arg_parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)

    opt = arg_parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2    #test

    # Training set
    train_src_word_inststance = read_instances_from_file(opt.train_src, opt.max_word_seq_len)
    train_tgt_word_inststance = read_instances_from_file(opt.train_tgt, opt.max_word_seq_len)

    if len(train_src_word_inststance) != len(train_tgt_word_inststance):
        print('warn: If training instance count is not equal!')
        quit()

    # validation set
    valid_src_word_instance = read_instances_from_file(opt.valid_src, opt.max_word_seq_len)
    valid_tgt_word_instance = read_instances_from_file(opt.valid_tgt, opt.max_word_seq_len)

    if len(valid_src_word_instance) != len(valid_tgt_word_instance):
        print('warn: If validation instance count is not equal')
        quit()

    # vocabulary
    src_word_to_idx = json.load(open(opt.src_vocab)) # inter data exchange
    tgt_word_to_idx = json.load(open(opt.tgt_vocab))

    # word to index
    print('info: convert source word to sequence of words')
    train_src_instance = convert_instance_to_idx_seq(train_src_word_inststance, src_word_to_idx)
    valid_src_instance = convert_instance_to_idx_seq(valid_src_word_instance, src_word_to_idx)

    print('info: convert target word to sequence of words.')
    train_tgt_instance = convert_instance_to_idx_seq(train_tgt_word_inststance, tgt_word_to_idx)
    valid_tgt_instance = convert_instance_to_idx_seq(valid_tgt_word_instance, tgt_word_to_idx)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word_to_idx,
            'tgt': tgt_word_to_idx},
        'train': {
            'src': train_src_instance,
            'tgt': train_tgt_instance},
        'valid': {
            'src': valid_src_instance,
            'tgt': valid_tgt_instance}}

    print('info: save all processed data into a file', opt.save_data)
    torch.save(data, opt.save_data)
    print('---done---')

if __name__ == '__main__':
	main()