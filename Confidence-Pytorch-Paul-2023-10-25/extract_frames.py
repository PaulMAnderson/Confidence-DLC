import deeplabcut
import os

# Define paths and parameters
config_path = '/mnt/c/_work_panderson/Analysis/Deeplabcut/Confidence-Pytorch-Paul-2023-10-25/tf_config.yaml'

# Create a new video folder path (optional) 
video_path = '/mnt/c/_work_panderson/Analysis/Deeplabcut/Confidence-Pytorch-Paul-2023-10-25/videos'  # Folder containing videos to analyze

# Extract frames using DeepLabCut's built-in function
deeplabcut.extract_frames(
    config=config_path,
    mode='automatic',  # Can be 'automatic', 'manual', or 'kmeans'
    algo='uniform',    # 'uniform' or 'kmeans'
    crop=False,        # Set True if you want to crop frames
    userfeedback=True,
    videos_list=['/mnt/c/_work_panderson/Analysis/Deeplabcut/Confidence-Pytorch-Paul-2023-10-25/videos/RMM55 2025_04_15_12_37_20.mkv']  # Can be a list of videos
)
