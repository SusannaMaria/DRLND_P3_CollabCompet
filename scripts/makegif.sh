#!/bin/bash
# Should work on Ubuntu 20.04, but please install
# sudo apt-get install kazam gifsicle imagemagick before trying
# Screencast done with kazam

mkdir -p gifs
rm -rf gifs/*
ffmpeg -i Kazam_screencast_00000.avi gifs/out1%04d.png
ffmpeg -i Kazam_screencast_00001.avi gifs/out2%04d.png

pushd gifs
mogrify -background SkyBlue4 -alpha remove -alpha off *.png
mogrify -resize 400x300 *.png
mogrify -format gif *.png
gifsicle --delay=10 --colors 256 --loop *.gif  > ../game.gif

rm -rf gifs