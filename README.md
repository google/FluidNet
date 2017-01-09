FluidNet 
============

![alt text](https://github.com/google/FluidNet/blob/master/sample.png "Sample png")

This repo contains all the code required to replicate the paper:

[Accelerating Eulerian Fluid Simulation With Convolutional Networks, Jonathan Tompson, Kristofer Schlachter, Pablo Sprechmann, Ken Perlin](http://cims.nyu.edu/~schlacht/CNNFluids.htm).

The workflow is: 

1. Generate the training data (simulation takes a few days for the 3D dataset).
    - Download the models and voxelize them.
    - Run mantaflow to generate fluid data.
2. Train the network in Torch7 (training takes about 1-2 days).
    - Train and validate the model.
    - (optional) Run 3D example script to create videos from the paper.
    - (optional) Run 2D real-time demo.

The limitations of the current system are outlined at the end of this doc. Please read it before even considering integrating our code.

Note: This is not an official Google product.

UPDATES / NEWS:
---------------

**Jan 6 2017**
- Lots of updates to model and training code.
- Added model of Yang et al. "Data-driven projection method in fluid simulation" as a baseline comparison.
- Changed model defaults (no more pooling, smaller network no more p loss term).
- Improved programmability of the model from the command line.
- Added additional scripts to plot debug data and performance vs. epoch.

**Dec 21 2016** 

- Refactor of data processing code.
- Batch creation is now asynchronous and parallel (to hide file IO latency). Results in slight speed up in training for most systems, and significant speedup for disk IO limited systems (i.e. when files are on a DFS).
- Data cache paths are now relative, so that cache data can be moved around.
- Implemented advection in CUDA; entire simulation.lua loop is now on the GPU. Significant speedup for 3D models (both training and eval) and slight speedup for 2D models.
- Numerous bug fixes and cleanup.

#0. Clone this repo:
--------------------

```
git clone git@github.com:google/FluidNet.git
```

#1. Generating the data
-------------------
**CREATING VOXELIZED MODELS**

We use a subset of the NTU 3D Model Database models (http://3d.csie.ntu.edu.tw/~dynamic/database/). Please download the model files:

```
cd FluidNet/voxelizer
mkdir objs
cd objs
wget http://3d.csie.ntu.edu.tw/~dynamic/database/NTU3D.v1_0-999.zip
unzip NTU3D.v1_0-999.zip
wget https://www.dropbox.com/sh/5f3t9abmzu8fbfx/AAAkzW9JkkDshyzuFV0fAIL3a/bunny.capped.obj
```

Next we use the binvox library (http://www.patrickmin.com/binvox/) to create voxelized representations of the NTU models. Download the executable for your platform and put the ``binvox`` executable file in ``FluidNet/voxelizer``. Then run our script:

```
cd FluidNet/voxelizer
chmod u+x binvox
python generate_binvox_files.py
```

Note: some users have reported that they need to install ``lib3ds-1-3``:

```
sudo apt-get install lib3ds-1-3
```

OPTIONAL: You can view the output by using the viewvox utility (http://www.patrickmin.com/viewvox/). Put the viewvox executable in the ``FluidNet/voxelizer/voxels`` directory, then:

```
cd FluidNet/voxelizer/voxels
chmod u+x viewvox
./viewvox -ki bunny.capped_32.binvox
```

**BUILDING MANTAFLOW**

The first step is to download the [custom manta fork](https://github.com/kristofe/manta).

```
cd FluidNet/
git clone git@github.com:kristofe/manta.git
```

Next, you must build mantaflow using the cmake system.

```
cd FluidNet/manta
mkdir build
cd build
sudo apt-get install doxygen libglu1-mesa-dev mesa-common-dev qtdeclarative5-dev qml-module-qtquick-controls
cmake .. -DGUI='OFF' 
make -j8
```

For the above cmake command setting ``-DGUI='ON'`` will slow down simulation but you can view the flow fields. You will now have a binary called manta in the build directory.

**GENERATING TRAINING DATA**

Install matlabnoise (https://github.com/jonathantompson/matlabnoise) to the SAME path that FluidNet is in. i.e. the directory structure should be:

```
/path/to/FluidNet/
/path/to/matlabnoise/
```

To install matlabnoise (with python bindings):

```
sudo apt-get install python3.5-dev
sudo apt-get install swig
git clone git@github.com:jonathantompson/matlabnoise.git
cd matlabnoise
sh compile_python3.5_unix.sh
sudo apt-get install python3-matplotlib
python3.5 test_python.py
```

Now you're ready to generate the training data. Make sure the directory `data/datasets/output_current` exists. For the 3D training data run:

```
cd FluidNet/manta/build
./manta ../scenes/_trainingData.py --dim 3 --addModelGeometry True --addSphereGeometry True
```

For the 2D data run:

```
cd FluidNet/manta/build
./manta ../scenes/_trainingData.py --dim 2 --addModelGeometry True --addSphereGeometry True
```

#2. Training the model
----------------------

**RUNNING TORCH7 TRAINING**

We assume that Torch7 is installed, otherwise follow the instructions [here](
http://torch.ch/). We use the standard [distro](https://github.com/torch/distro) with the cuda SDK for cutorch and cunn and [cudnn](https://github.com/soumith/cudnn.torch).

After install torch, compile tfluids: this is our custom CUDA & C++ library that implements a large number of the modules used in the paper:

```
sudo apt-get install freeglut3-dev
sudo apt-get install libxmu-dev libxi-dev
cd FluidNet/torch/tfluids
luarocks make tfluids-1-00.rockspec
```

Note: some users are reporting that you need to explicitly install findCUDA for tfluids to compile properly with CUDA 7.5 and above.

```
luarocks install findCUDA
```

All training related code is in `torch/` directory.  To train a model on 3D data:

```
cd FluidNet/torch
qlua fluid_net_train.lua -gpu 1 -dataset output_current_3d_model_sphere -modelFilename myModel3D
```

This will pull data from the directory `output_current_3d_model_sphere` and dump the model to `myModel3D`. To train a 2D model:

```
cd FluidNet/torch
qlua fluid_net_train.lua -gpu 1 -dataset output_current_model_sphere -modelFilename myModel2D
```

At any point during the training sim you can plot test and training set loss 
values using the Matlab script `FluidNet/torch/utils/PlotEpochs.m`.

You can control any model or training config parameters from the command line. 
If you need to define nested variables the syntax is:

```
qlua fluid_net_train.lua -new_model.num_banks 2
```

i.e nested variables are `.` separated. You can print a list of possible 
config variables using:

```
qlua fluid_net_train.lua --help
```

Note: the first time the data is loaded from the manta output, it is cached to 
the torch/data/ directory.  So if you need to reload new data (because you altered the dataset) then delete the cache files (`torch/data/*.bin`).

**RUNNING THE REAL-TIME DEMO**

For 2D models only! To run the interactive demo firstly compile LuaGL:

```
git clone git@github.com:kristofe/LuaGL.git
cd LuaGL
luarocks make luagl-1-02.rockspec
```

Then run the simulator:

```
cd FluidNet/torch
qlua -lenv fluid_net_2d_demo.lua -gpu 1 -dataset output_current_model_sphere -modelFilename myModel2D
```

The command line output will print a list of possible key and mouse strokes.

**RUNNING THE 3D SIMULATIONS**

To render the videos you will need to install [Blender](http://www.blender.org), but to just create the volumetric data no further tools are needed. First run our 3D example script (after training a 3D model):

```
cd FluidNet/torch
qlua fluid_net_3d_sim.lua -gpu 1 -loadVoxelModel none -modelFilename myModel3D
```

To control which scene is loaded, use the ``loadVoxelModel="none"|"arc"|"bunny"```. This will dump a large amount of volumetric data to the file ``FluidNet/blender/<mushroom_cloud|bunny|arch>_render/output_density.vbox``.

Now that the fluid simulation has run, you can render the frames in Blender. Note that rendering takes a few hours, while the 3D simulation is fast (with a lot of time spent dumping the results to disk). An implementation of a real-time 3D fluid render is outside the scope of this work. In addition, self-advection of the velocity field is currently carried out on the CPU and so is the slowest part of our simulator (a CUDA implementation is future work).

For the mushroom cloud render, open ``FluidNet/blender/MushroomRender.blend``. Next we need to re-attach the data file (because blender caches full file paths which will now be wrong). Click on the "Smoke" object in the "Outliner" window (default top right). Click on the "Texture" icon in the "Properties" window (default bottom right), it's the one that looks like a textured Square. Scroll down to "Voxel Data" -> "Source Path:" and click the file icon. Point the file path to ``/path/to/FluidNet/blender/mushroom_cloud_render/density_output.vbox``. Next, click either the file menu "Render" -> "Render Image", or "Render Animation". By default the render output goes to ``/tmp/``.  You can also scrub through the frame index on the time-line window (default bottom) to click a frame you want then render just that frame.

The above instructions also apply to the bunny and arch examples. Note: you might need to re-center the model depending on your version of binvox (older versions of binvox placed the voxelided model in a different location). If this is the case, then click on the "GEOM" object in the "Outliner" window. Click the eye and camera icons (so they are no longer greyed out). Then press "Shift-Z" to turn on the geometry render preview. Now that you can see the geometry and model, you can manually align the two so they overlap.

# 3. Limitations of the current system
--------------------------------------

While this codebase is relatively self-contained and full-featured, **it is not a "ready-to-ship" fluid simluator**. Rather it is a proof of concept and research platform only. If you are interested in integrating our network into an existing system feel free to reach out (tompson@google.com) and we will do our best to answer your questions.

**RUNTIME**

The entire simulation loop is not optimized; however it is fast enough for real-time applications, where good GPU resources are available (i.e. NVidia 1080 or Titan).

**BOUNDARY HANDLING**

Our example boundary condition code is very rudimentary. However, we support the same cell types as Manta (in-flow, empty, occupied, etc), so more complicated boundary conditions can be created. One potential limitation is that the setWallBcs codepath assumes zero velocity occupiers (like Manta does). However, it would be an easy extension to allows internal occupied voxels to have non-zero velocity.

**RENDERING**

We do not have a real-time 3D fluid render. We use an offline render instead. For our 2D "renderer", we simply display the RGB density field to screen and visualize the velocity vectors. It is very rudimentary. Incorporating an open-source 3D fluid render is future work.

**SIMULATOR**

The only external forces that are supported are vorticity confinement and buoyancy. Viscosity and gravity are not supported (but could be added easily).

**UNIT TESTING**

We have unit tests (including FD gradient checks) for all custom torch modules. 

The two main test scripts we do have are:

```
cd FluidNet/
qlua lib/modules/test_ALL_MODULES.lua
```

and (this one requires us to generate data from manta first):

```
cd FluidsNet/manta/build
./manta ../scenes/_testData.py
cd ../../torch/tfluids
qlua -ltfluids -e "tfluids.test()"
```

You should run these first if you ever get into trouble training or running the model.
