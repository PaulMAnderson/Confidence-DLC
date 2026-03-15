import deeplabcut

# Path to your existing project config file
config_path = '/mnt/c/_work_panderson/Analysis/Deeplabcut/Confidence-Pytorch-Paul-2023-10-25/tf_config.yaml'

# Create a new PyTorch-based project from your existing project
deeplabcut.create_training_dataset(config_path, net_type='resnet_50', augmenter_type='imgaug')

# Edit the configuration file to use PyTorch backend
# This can be done automatically with:
deeplabcut.auxiliaryfunctions.edit_config(config_path, {'dependency': 'pytorch'})

# Train the model using the PyTorch backend
deeplabcut.train_network(config_path, shuffle=1, gputouse=0, max_snapshots_to_keep=5)
