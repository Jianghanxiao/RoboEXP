conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install open_clip_torch
pip install --upgrade openai
pip install opencv-python
pip install transforms3d
pip install open3d
# Install SegmentAnything and GroundingDINO
pip install git+https://github.com/IDEA-Research/Grounded-Segment-Anything.git#subdirectory=segment_anything 
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
conda install python-graphviz

# Install the xarm python sdk
pip install git+https://github.com/xArm-Developer/xArm-Python-SDK.git

# Install the realsense2
pip install pyrealsense2