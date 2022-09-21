import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--tiny_sample", dest="tiny_sample", action="store_true") 
parser.add_argument("--num_prototypes", type=int, default=20)
parser.add_argument("--num_pos_prototypes", type=int, default=19)


parser.add_argument("--model", type=str, default="ProtoTEx")



args = parser.parse_args()