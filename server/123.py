import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--new_model", help="if initialize a new model or not, [default]:False",
                   default = False, required=False)
args = parser.parse_args()
print(vars(args))
