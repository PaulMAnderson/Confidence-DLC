import glob
import os
import deeplabcut

config_path =  '/mnt/c/_work_panderson/Analysis/Deeplabcut/Confidence-Pytorch-Paul-2023-10-25/tf_config.yaml'

inputPath = '/mnt/c/_work_panderson/Analysis/Deeplabcut/Confidence-Pytorch-Paul-2023-10-25/test videos'
videoPath = os.path.abspath(inputPath)
videoList = glob.glob(videoPath + '/*.mp4')

print('inputPath = ', inputPath)
print('videoPath = ', videoPath)
print('videoList = ', videoList)

# Trying command with 'allow growth' should limit gpu ussage
deeplabcut.analyze_videos(config_path, videoList, videotype='.mp4', shuffle=1, trainingsetindex=0, save_as_csv=True, allow_growth=False)

deeplabcut.filterpredictions(config_path,[videoPath], videotype='.mp4',filtertype='median',windowlength=9)


deeplabcut.create_labeled_video(
    config_path, [videoPath], filtered=True, trailpoints=10, fastmode=False, dotsize=15
    )