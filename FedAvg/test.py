import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rounds", help="aggregation rounds",
                    type=int)
args = parser.parse_args()

rounds = args.rounds
if not rounds:
    raise ValueError('Plz type in aggregation rounds')
print(rounds)

