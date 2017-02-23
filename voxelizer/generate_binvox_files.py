# Copyright 2016 Google Inc, NYU.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from subprocess import call
import random

def get_file_list_all_files(folder, suffix):
  fileList = []
  for i in os.listdir(folder):
    if i.endswith(suffix):
      fileList.append(i)
  return fileList


def get_file_list(txtfile):
  with open(txtfile, 'r') as f:
    return f.read().splitlines()


def create_binvox_file(inputFilename, inputFolder, outputFolder, size):
  f = inputFolder + inputFilename
  callStr = "./binvox -d "+str(size) + " " + f
  print("Calling %s" % callStr)
  call(callStr, shell=True)

# File now exist with binvox extenstion in inputFilename folder.
# Remove ".obj" extension and create correct input and output filenames.
  voxelFile = inputFilename[:-4]
  srcVoxelFilePath = inputFolder + voxelFile + ".binvox"
  dstVoxelFilePath = outputFolder + voxelFile + "_" + str(size) + ".binvox"
  
# move file to voxels directory with modified filename to reflect resolution.
  callStr = "mv " + srcVoxelFilePath  + " " + dstVoxelFilePath
  call(callStr, shell=True)

if __name__ == "__main__":
  inputFolder = os.getcwd() + "/objs/"
  suffix = ".obj"
  fileList = get_file_list("obj_files.txt")

  random.seed(0)
  random.shuffle(fileList)  # Note: random is already seeded above.

  # We need higher res binvox files for the arc and bunny models (because we
  # will use these for our demo simulations).
  outputFolder = os.getcwd() + "/voxels_demo/"
  if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
  sizes = [8, 16, 32, 64, 128, 256]
  for i in range(len(sizes)):
    create_binvox_file("Y91_arc.obj", inputFolder, outputFolder, sizes[i])
    create_binvox_file("bunny.capped.obj", inputFolder, outputFolder, sizes[i])

  # Now create the test and training sets.
  sizes = [8, 16, 32]
  total = len(fileList) * len(sizes)
  pct = 0.0
  count = 0.0
  numTraining = int(len(fileList) / 2)
  for ind in range(len(fileList)):
    f = fileList[ind]
    if ind < numTraining:
      outputFolder = os.getcwd() + "/voxels_train/"
    else:
      outputFolder = os.getcwd() + "/voxels_test/"
    if not os.path.exists(outputFolder):
      os.makedirs(outputFolder)
    for size in sizes:
      count = count + 1
      print "%s - %d. Percent Complete: %.1f" % (f, size, 100.0 * (count/total))
      create_binvox_file(f, inputFolder, outputFolder, size)
