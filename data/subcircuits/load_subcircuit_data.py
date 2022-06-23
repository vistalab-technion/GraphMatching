import pickle
import os

filename = os.getcwd() +'/ibex_lsi_10k_synopsys_masks/ibex_lsi_10k_synopsys.p'
file = open(filename, 'rb')
circuit = pickle.load(file)
file.close()

