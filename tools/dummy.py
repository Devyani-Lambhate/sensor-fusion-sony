import torch

ckpt_path = "/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet_vishal/output/nuscenes_models/bevfusion/fog_finetuned_mixed/ckpt/final_model_epoch_1.pth"

# Load checkpoint
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only = False)

# Print top-level keys
print("Top-level keys:")
print(checkpoint.keys())

model_state = checkpoint.get("model_state", None)
print("\nModel state keys:")
if model_state is not None:
    print(model_state.keys())

stats = checkpoint.get("stats", {})
print("\nStats keys:")
print(stats.keys()) 

sigma = checkpoint.get("sigmas_history", None)
print("\nSigma:", sigma)
if sigma is not None:
    print("\nSigma shape:", len(sigma)) 

full_uncertainties = checkpoint.get("full_uncertainties", None)
print("\nFull uncertainties keys:")
if full_uncertainties is not None:
    print(full_uncertainties.keys())

center_scale = full_uncertainties.get("center_scale", None)
print("\nCenter scale shape:", center_scale.shape if center_scale is not None else None)

num_predictions = checkpoint.get("num_predictions", None)
print("\nNumber of predictions:", num_predictions)  

print("\n" + "="*50)

# Optional: recursively print nested keys
def print_keys(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{prefix}{k}")
            print_keys(v, prefix + "    ")

print("\nAll nested keys:")
print_keys(checkpoint)