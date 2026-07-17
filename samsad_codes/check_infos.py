import pickle

info_files = '/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet/data/nuscenes-trainval/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl'

with open(info_files, 'rb') as f:
    infos = pickle.load(f)

print(f"Type of infos: {type(infos)}")
print(f"Length of infos: {len(infos)}")
print(f"Type of each info: {type(infos[0]) if len(infos) > 0 else 'N/A'}")

# Dict keys
print(f"Keys of each info: {list(infos[0].keys()) if len(infos) > 0 else 'N/A'}")
