import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--contact_label_dir", type=str, default="", help="Path to the contact label dataset directory")
parser.add_argument("--motion_data_dir", type=str, default="", help="Path to the motion data directory")
parser.add_argument("--output_dir", type=str, default="", help="Path to save the merged dataset")
args = parser.parse_args()

# load contact dataset
contact_label_dir = args.contact_label_dir
data_files = os.listdir(contact_label_dir)
contact_data = {}
for file in data_files:
    if file.endswith(".npy"):
        data = np.load(os.path.join(contact_label_dir, file))
        raw_key = file.split(".")[0]  # like sub17_small_box_1
        # formate to sub17_small_box_001
        formatted_key = raw_key.rsplit("_", 1)[0] + "_" + raw_key.rsplit("_", 1)[1].zfill(3)
        contact_data[formatted_key] = data

print(f"Loaded contact label dataset from {contact_label_dir}, total {len(contact_data)} files.")

# load corresponding motion dataset
data_file_keys = list(contact_data.keys())
motion_data_dir = args.motion_data_dir
motion_files = os.listdir(motion_data_dir)
motion_data = {}
for file in motion_files:
    if file.endswith(".npz"):
        raw_key = file.split(".")[0][13:] # like sub17_small_box_1
        # formate to sub17_small_box_001
        #formatted_key = raw_key.rsplit("_", 1)[0] + "_" + raw_key.rsplit("_", 1)[1].zfill(3)
        if raw_key not in data_file_keys:
            continue
        data = np.load(os.path.join(motion_data_dir, file))
        motion_data[raw_key] = dict(data)

# interpolate contact labels to match motion data length
for key in data_file_keys:
    contact_labels = contact_data[key]  # shape [T_contact]
    motion_length = motion_data[key]["body_pos_w"].shape[0]  # T_motion
    contact_length = contact_labels.shape[0]

    if motion_length == contact_length:
        interpolated_labels = contact_labels
    else:
        interpolated_labels = np.zeros(motion_length, dtype=contact_labels.dtype)
    
        interpolated_labels = np.interp(
            np.linspace(0, contact_length - 1, motion_length),
            np.arange(contact_length),
            contact_labels
        ).round().astype(contact_labels.dtype)

    # save back to motion data
    motion_data[key]["wrist_grasp_label"] = interpolated_labels

# save merged dataset
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
for key in data_file_keys:
    output_path = os.path.join(output_dir, f"motion_{key}.npz")
    np.savez_compressed(output_path, **motion_data[key])

print(f"Merged dataset saved to {output_dir}, total {len(motion_data)} files.")