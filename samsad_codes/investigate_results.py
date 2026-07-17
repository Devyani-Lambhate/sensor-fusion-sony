import pickle
import os

info_pkl = "data/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl"
result_pkl = "output/nuscenes_models/bevfusion/default/eval/epoch_no_number/val/original_dataset/result.pkl"


with open(info_pkl, "rb") as f:
    infos = pickle.load(f)

with open(result_pkl, "rb") as f:
    results = pickle.load(f)

info_tokens = [x["token"] for x in infos]
result_tokens = [x["metadata"]["token"] for x in results]

print("Length checks")
print("infos:", len(info_tokens))
print("results:", len(result_tokens))

print("\nUniqueness checks")
print("unique info tokens:", len(set(info_tokens)))
print("unique result tokens:", len(set(result_tokens)))

info_set = set(info_tokens)
result_set = set(result_tokens)

print("\nSet equality")
print("Exact match:", info_set == result_set)

missing_in_results = info_set - result_set
extra_in_results = result_set - info_set

print("\nMismatch summary")
print("Missing in results:", len(missing_in_results))
print("Extra in results:", len(extra_in_results))

if missing_in_results:
    print("\nSample missing tokens:")
    print(list(missing_in_results)[:10])

if extra_in_results:
    print("\nSample extra tokens:")
    print(list(extra_in_results)[:10])

print("\nOrder check")
print("Same order:", info_tokens == result_tokens)

if info_tokens != result_tokens:
    print("\nFirst mismatch")
    for i, (itok, rtok) in enumerate(zip(info_tokens, result_tokens)):
        if itok != rtok:
            print("Index:", i)
            print("Info token:", itok)
            print("Result token:", rtok)
            break 