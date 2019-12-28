import os
import argparse
import h5py

# Get the files names
parser = argparse.ArgumentParser(description="Train a classifier for fingerprints")
parser.add_argument("in1", help="Input first database file path")
parser.add_argument("in2", help="Input second database file path")
args = vars(parser.parse_args())
filename1 = args["in1"]
filename2 = args["in2"]
# Open both files and append the content of the second to the first
with h5py.File(filename1, 'a') as f1, h5py.File(filename2, 'r') as f2:
	print("Output file", filename1, "and input file", filename2, "opened")
	for name in f2.keys():
		if not name in f1.keys():
			print("The", name, "dataset does not exist in the output file")
		else:
			# Open input database and read attributes
			idb = f2[name]
			num_classes = idb.attrs['num_classes']
			irep = idb.attrs['repetitions']
			# Open the database with given name and check attributes
			db = f1[name]
			if (not 'num_classes' in db.attrs) or db.attrs['num_classes'] != num_classes:
				raise ValueError("The dataset lacks 'num_classes' or it differs from input folder traits")
			if (not 'repetitions' in db.attrs):
				raise ValueError("The dataset lacks 'repetitions'")
			prev_rep = db.attrs['repetitions']
			db.attrs['repetitions'] += irep
			db.resize((db.shape[0]+irep*num_classes, db.shape[1]))
			print("Appending logits for", irep, "repetitions of", num_classes, "classes to dataset", name, "(total "+str(db.attrs['repetitions'])+" repetitions)")
			begin = prev_rep * num_classes # inclusive
			end = db.attrs['repetitions'] * num_classes # exclusive
			db[begin:end, :] = idb[:, :]
