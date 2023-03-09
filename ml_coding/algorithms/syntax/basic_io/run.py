import argparse
import datetime
import os
import logging
import sys
from typing import Any, cast, Dict, List, Tuple, Union


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main() -> None:
	
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--source",
		default="src.txt", # could do required=True instead.
		type=str,
		help= "source file name.")
	args = parser.parse_args()
	source = args.source

	abs_path = os.path.realpath(__file__)
	curr_dir = os.path.dirname(abs_path) # same as os.getcwd()
	parent_dir = os.path.dirname(curr_dir)
	new_dir = os.path.join(curr_dir, "dest")
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)

	timestamp = datetime.datetime.now()
	src_path = os.path.join(curr_dir, source)
	dst_path = os.path.join(new_dir, f"dst_{timestamp.date()}_{timestamp.time()}.txt")
	with open(src_path, "r") as f:
		s = f.read()
	with open(dst_path, "w") as f:
		f.write(s)




if __name__ == "__main__":
	main()