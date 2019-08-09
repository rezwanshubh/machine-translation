import argparse

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

if __name__ == '__main__':
	main()