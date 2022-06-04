import pickle

filename = 'ibex_lsi_10k_synopsys.p'
file = open(filename, 'rb')
circuit = pickle.load(file)
file.close()

filename = 'ibex_lsi_10k_synopsys.p'
