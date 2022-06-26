import pickle
import os

filename = os.getcwd() +'/ibex_lsi_10k_synopsys_masks/ibex_lsi_10k_synopsys.p'
file = open(filename, 'rb')
circuit = pickle.load(file)
file.close()


filename = os.getcwd() +'/ibex_lsi_10k_synopsys_masks/core.p'
file = open(filename, 'rb')
core = pickle.load(file)
file.close()