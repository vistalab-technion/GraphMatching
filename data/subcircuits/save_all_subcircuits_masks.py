import pickle
from APIImplementation.NetlistSmoothening.DependencyFrameGenerator import load_frame
from Configuration.Config import Config
from APIImplementation.Cluttering.Mask import get_subcircuit_names, get_all_circuit_subcircuit_masks


chosen_netlist_name = "ibex_lsi_10k_synopsys"
output_folder_relative_path = f"./{chosen_netlist_name}_masks/"
chosen_netlist_full_name = f"FF_dependencies__{chosen_netlist_name}.v.csv"
compund_circuit_frame_path = Config.auto_generated_saved_frames_directory + chosen_netlist_full_name
# compund_circuit_frame_path = Config.real_netlists_saved_frames_directory + chosen_netlist_full_name

compund_circuit_frame = load_frame(compund_circuit_frame_path)
subcircuit_names = list(get_subcircuit_names(compund_circuit_frame))

flag, masks = get_all_circuit_subcircuit_masks(compund_circuit_frame, subcircuit_names[0])
if not flag:
    exit(-1)

circuit_matrix, subcircuit_name_to_mask_map = masks

circuit_mask_file_path = f"{output_folder_relative_path}/{chosen_netlist_name}.p"
with open(circuit_mask_file_path, 'wb') as f:
    pickle.dump(circuit_matrix, f)


for subcircuit_name, subcircuit_mask in subcircuit_name_to_mask_map.items():
    subrcircuit_mask_file_path = f"{output_folder_relative_path}./{subcircuit_name.replace('/', '')}.p"
    with open(subrcircuit_mask_file_path, 'wb') as f:
        pickle.dump(subcircuit_mask, f)
