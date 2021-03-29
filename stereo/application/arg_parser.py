# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-23

import argparse

parser = argparse.ArgumentParser(description='Train a stereo model.')

# Working directory and naming.
parser.add_argument("--working-dir", type=str, default="./Debug", \
    help="The working directory.")

parser.add_argument("--model-name", type=str, default="Stereo", \
    help="The name of the model. ")

parser.add_argument("--prefix", type=str, default="", \
    help="The prefix of the work flow. The user should supply delimiters such as _ .")

parser.add_argument("--suffix", type=str, default="", \
    help="The suffix o fthe work flow. The user should supply delimiters such as _ .")

# Model & optimizer IO.
parser.add_argument("--read-model", type=str, default="", \
    help="Read model from working directory. Supply empty string for not reading model.")

parser.add_argument("--read-optimizer", type=str, default="", \
    help="Read the optimizer state from the working directory. Leave blank for not reading the optimizer.")

# Hardware settings.
parser.add_argument("--multi-gpus", action="store_true", default=False, \
    help="Use multiple GPUs.")

parser.add_argument("--cpu", action="store_true", default=False, \
    help="Set this flag to use cpu only. This will overwrite --multi-gpus flag.")

# Data loader.
parser.add_argument("--data-json-list", type=str, default="", \
    help="The file contains a list of dataset json files. Cannot used together with --data-root-dir.")

parser.add_argument("--train-epochs", type=int, default=10, \
    help="The number of training epochs.")

# Testing.
parser.add_argument("--test", action="store_true", default=False, \
    help="Only perform test. Make sure to specify --read-model")

parser.add_argument("--test-loops", type=int, default=0, \
    help="The number of training loops between a test. Set 0 for not testing.")

parser.add_argument("--test-flag-save", action="store_true", default=False, \
    help="Set this flag to save the test result as images and disparity files.")

# Logging.
parser.add_argument("--train-interval-acc-write", type=int, default=10, \
    help="Write the accumulated data to filesystem by the number of loops specified.")

parser.add_argument("--train-interval-acc-plot", type=int, default=1, \
    help="Plot the accumulated data to filesystem by the number of loops specified.")

parser.add_argument("--use-intermittent-plotter", action="store_true", default=False, \
    help="Use the intermittent plotter instead of the Visdom plotter. NOTE: Make sure to set --train-interval-acc-plot accordingly.")

parser.add_argument("--auto-save-model", type=int, default=0, \
    help="The number of loops to perform an auto-save of the model. 0 for disable auto-saving.")

parser.add_argument("--auto-snap-loops", type=int, default=100, \
    help="The number of loops for auto-snap.")

parser.add_argument("--disable-stream-logger", action="store_true", default=False, \
    help="Disable the stream logger of WorkFlow.")

args = parser.parse_args()
