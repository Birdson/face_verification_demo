#!/bin/bash

# use config file
config_path="config.ini"
if [ ! -f $config_path ]; then
    echo "Require $config_path , Exit"
    exit
fi

function read_variable () {
    echo $(sed -n "s/.*$1 *= *\([^ ]*.*\)/\1/p" < $config_path)
}

enable_virtual_device=$(read_variable enable_virtual_device)

if [ "$enable_virtual_device" = "true" ]; then
    sudo modprobe v4l2loopback exclusive_caps=1
fi

./build/face_verification_demo
