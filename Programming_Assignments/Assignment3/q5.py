import utils
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='q5')
    parser.add_argument('-q', help='q parameter', required=True)
    args = parser.parse_args()
    num_seq = 10000
    # Check if given sequence of digit created based on q parameter is already produced
    # otherwise create it
    filepath = Path("data/digits_num_seq=" + str(num_seq) + "_q=" + str(args.q) + ".csv")
    if not filepath.exists():
        utils.create_sequence_digits(q=args.q, num_seq=num_seq)
    else:
        # Question 5a
        utils.uniformity_test_5(args.q, delta=0.1, case="single digit")
        # Question 5b
        utils.uniformity_test_5(args.q, delta=0.1, case="pairs")
        utils.uniformity_test_5(args.q, delta=0.1, case="triples")
