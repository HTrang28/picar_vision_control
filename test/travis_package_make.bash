#!/bin/bash -xve

sudo apt-get update
sudo apt-get install libopencv-core-dev libopencv-dev python-opencv

#sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
#sudo ldconfig

#sudo apt-get install libopencv-dev
sudo apt-get install ros-kinetic-cv-bridge
sudo apt-get install ros-kinetic-cv-camera

#sync and make
rsync -av ./ ~/catkin_ws/src/pimouse_vision_control/

#clone pimouse_ros
cd ~/catkin_ws/src/
git clone https://github.com/citueda/pimouse_ros.git

cd ~/catkin_ws
catkin_make

# Copyright 2016 Ryuichi Ueda
# Released under the MIT License.
# To make line numbers be identical with the book, this statement is written here. Don't move it to the header.
