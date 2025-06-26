import nibabel as nib
from tqdm import tqdm
import os

from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    SpatialPadd,
    LambdaD,
    EnsureChannelFirstd,
  )
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.networks.nets import SwinUNETR
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

import torch

import argparse
import sys
import csv
import numpy as np

###############################################################################################################
# Argument definitions
###############################################################################################################
parser = argparse.ArgumentParser(description="Train a MONAI model with arguments")

# required arguments
parser.add_argument("-exp_name", type=str, required=True, help="Name of experiment")
parser.add_argument("-json_dir", type=str, required=True, help="Path of json")

# optional arguments
parser.add_argument("-max_iterations", type=int, default=30000, help="Maximum number of training iterations")
parser.add_argument("-size_patch", nargs=3, type=int, default=[96,96,96], help="Spatial size of training patches (x, y, z)")
parser.add_argument("-dim_voxel", nargs=3, type=float, default=[1.5,1.5,2.0], help="Voxel spacing in millimeters (x, y, z)")
parser.add_argument("-size_batch", type=int, default=1, help="Batch size used for training and validation")
parser.add_argument("-size_cache_test", type=int, default=6, help="Number of test samples to cache in memory")
parser.add_argument("-channels_out", type=int, default=14, help="Number of output channels (e.g., number of classes)")
parser.add_argument("-prob_drop", type=float, default=0.0, help="Dropout probability for regularization")
parser.add_argument("-postfix", type=str, default="", help="Postfix to append to the experiment name")
parser.add_argument("-intensity_range", nargs=2, type=float, default=[-175, 250], help="Range for input intensity normalization")
parser.add_argument("-fg_class", nargs='+', type=int, default=[1], help="List of foreground class indices for segmentation")

# personalization mode
parser.add_argument("--personalize", action="store_true", help="Enable personalization mode with limited training samples")

args = parser.parse_args()

###############################################################################################################
# Apply CLI arguments to local variables
###############################################################################################################
exp_name = args.exp_name
max_iterations = args.max_iterations
size_patch = tuple(args.size_patch)
dim_voxel = tuple(args.dim_voxel)
size_batch = args.size_batch
size_cache_test = args.size_cache_test
channels_out = args.channels_out
json_dir = args.json_dir
personalize = args.personalize
prob_drop = args.prob_drop
postfix = args.postfix.strip()
a_min, a_max = args.intensity_range

###############################################################################################################
# Directory setup & script backup
###############################################################################################################
root_dir = './! history'
exp_path = os.path.join(root_dir, exp_name)

###############################################################################################################
# Transformation pipelines
###############################################################################################################
def make_fg_mask(class_list):
    # Returns a function that creates a foreground mask given a list of class indices
    def fg_func(label):
        # Create a boolean mask where any voxel matching the specified classes is True
        mask = torch.zeros_like(label, dtype=torch.bool)
        for c in class_list:
            mask |= (label == c)
        # Convert boolean mask to uint8 format (0 or 1)
        return mask.to(torch.uint8)
    return fg_func

test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image", "label"],channel_dim='no_channel'),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=dim_voxel,
            mode=("bilinear", "nearest"),
        ),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=size_patch,
            method="symmetric",
        ),
        EnsureTyped(keys=["image", "label"], device="cpu", track_meta=True),
        LambdaD(
            keys="label",
            func=make_fg_mask(args.fg_class)
        ),
    ]
)

###############################################################################################################
# Dataset & DataLoaders
###############################################################################################################
test_dicts = load_decathlon_datalist(json_dir, True, "test")
test_ds = CacheDataset(data=test_dicts, transform=test_transforms, cache_num=size_cache_test, cache_rate=1.0, num_workers=4)
test_loader = ThreadDataLoader(test_ds, num_workers=0, batch_size=size_batch)
 
# Logs
if personalize:
    log_file_path = os.path.join(exp_path, "test_overfit.log")
else:
    log_file_path = os.path.join(exp_path, "test.log")
log_file = open(log_file_path, "w")

sys.stdout = log_file
sys.stderr = log_file

set_track_meta(True)

###############################################################################################################
# Environment & device
###############################################################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

###############################################################################################################
# Model definition
###############################################################################################################
model = SwinUNETR(
    img_size=(size_patch, size_patch, size_patch),
    in_channels=1,
    out_channels=channels_out,
    feature_size=48,
    drop_rate = prob_drop,
    attn_drop_rate = prob_drop,
    # dropout_path_rate = prob_drop,
    use_checkpoint=True,
).to(device)

torch.backends.cudnn.benchmark = True

post_label = AsDiscrete(to_onehot=channels_out)
post_pred = AsDiscrete(argmax=True, to_onehot=channels_out)
dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)
hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False, percentile=95.0)
msd_metric = SurfaceDistanceMetric(include_background=False, reduction="mean", get_not_nans=False)

model.load_state_dict(torch.load(os.path.join(exp_path, "results/best_metric_model.pth")))    
postfix_suffix = f"_{postfix}" if postfix else ""
results_dir = os.path.join(exp_path, f"results_test{postfix_suffix}")
csv_path = os.path.join(results_dir, "test_dsc.csv")

def enable_dropout(model):
    """Force dropout to remain active even after model.eval()"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d) or isinstance(m, torch.nn.Dropout3d):
            m.train()


num_samples_mcd = 5
os.makedirs(results_dir, exist_ok=True)

# Initialize CSV file and write header  
with open(csv_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Case", "Uncertainty", "DSC", "HD95", "MSD"]) 

with torch.no_grad():
    epoch_iterator_val = tqdm(test_loader, desc="Test (X / X Steps) (dice=X.X)", dynamic_ncols=True)
    
    for case_idx, batch in enumerate(epoch_iterator_val):
        model.eval()
        enable_dropout(model)

        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
        outputs_mc = []
        for _ in range(num_samples_mcd):
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(
                    val_inputs, size_patch, 4, model, overlap=0.5
                )
            outputs_mc.append(val_outputs)
        outputs_mc = torch.stack(outputs_mc, dim=0)
        outputs_mc = torch.softmax(outputs_mc, dim=2)
        uncertainty_map = outputs_mc.std(dim=0)
        val_uncertainty = uncertainty_map[0,1,:,:,:].mean().item()

        model.eval()
        with torch.cuda.amp.autocast():
            val_outputs = sliding_window_inference(
                val_inputs, size_patch, 4, model, overlap=0.5
            )

        # Post-processing
        val_labels_list = decollate_batch(val_labels)
        val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
        val_outputs_list = decollate_batch(val_outputs)
        softmax_output = torch.softmax(val_outputs_list[0], dim=0)
        val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

        # Calculate metrics 
        dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        hd95_metric(y_pred=val_output_convert, y=val_labels_convert)
        msd_metric(y_pred=val_output_convert, y=val_labels_convert)
        
        case_dsc = dice_metric.aggregate().item()
        case_hd95 = hd95_metric.aggregate().item()
        case_msd = msd_metric.aggregate().item()

        dice_metric.reset()
        hd95_metric.reset()
        msd_metric.reset()

        # Save the results to a CSV file
        case_name = os.path.split(batch["image"].meta["filename_or_obj"][0])[1]
        with open(csv_path, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([case_name, val_uncertainty, case_dsc, case_hd95, case_msd])  

        # Save input 
        input_data = val_inputs[0, 0].detach().cpu().numpy()  
        affine = batch["image"].meta["affine"][0].cpu().numpy()
        input_nifti = nib.Nifti1Image(input_data, affine)
        input_path = os.path.join(results_dir, f"img{case_name.replace('.nii.gz', '').replace('img', '')}.nii.gz")
        nib.save(input_nifti, input_path)

        # Save label 
        label_data = val_labels[0, 0].detach().cpu().numpy()  
        label_nifti = nib.Nifti1Image(label_data, affine)
        label_path = os.path.join(results_dir, f"label{case_name.replace('.nii.gz', '').replace('img', '')}.nii.gz")
        nib.save(label_nifti, label_path)

        # Save probability 
        prob_volume = softmax_output.detach().cpu().numpy().astype(np.float32)
        prob_nifti = nib.Nifti1Image(prob_volume, affine)
        prob_path = os.path.join(results_dir, f"prob{case_name.replace('.nii.gz', '').replace('img', '')}.nii.gz")
        nib.save(prob_nifti, prob_path)

        # Save prediction 
        single_volume = torch.argmax(softmax_output, dim=0).detach().cpu().numpy().astype(np.uint8)
        prediction_nifti = nib.Nifti1Image(single_volume, affine)
        prediction_path = os.path.join(results_dir, f"pred{case_name.replace('.nii.gz', '').replace('img', '')}.nii.gz")
        nib.save(prediction_nifti, prediction_path)

        # Save MCD std map
        map_volume = uncertainty_map.detach().cpu().numpy().astype(np.float32)
        map_nifti = nib.Nifti1Image(map_volume, affine)
        map_path = os.path.join(results_dir, f"domap{case_name.replace('.nii.gz', '').replace('img', '')}.nii.gz")
        nib.save(map_nifti, map_path)

print(f"Test results saved to: {csv_path}")

# Note: Restore redirection to original state if necessary
log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__