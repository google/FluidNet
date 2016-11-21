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
    - (optional) Run 3D example script we used to generate the videos from our paper.
    - (optional) Run 2D real-time demo.

The limitations of the current system are outlined at the end of this doc. Please read it before even considering integrating our code.

Note: This is not an official Google product.

#0. Clone this repo:
--------------------

```
git clone git@github.com:jonathantompson/FluidNet.git
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
sudo apt-get install python3.4-dev
sudo apt-get install swig
git clone git@github.com:jonathantompson/matlabnoise.git
cd matlabnoise
sh compile_python3.4_unix.sh
python3.4 test_python.py
```

Now you're ready to generate the training data. Make sure the directory `data/datasets/output_current` exists. For the 3D training data run:

```
cd FluidNet/manta/build
./manta ../scenes/_trainingData.py --dim 3 --addGeometryType 'model'
```

For the 2D data run:

```
cd FluidNet/manta/build
./manta ../scenes/_trainingData.py --dim 2 --addGeometryType 'model'
```

#2. Training the model
----------------------

**RUNNING TORCH7 TRAINING**

We assume that Torch7 is installed, otherwise follow the instructions [here](
http://torch.ch/). We use the standard distro with the cuda SDK and cudnn. Note: there may be other libraries we use, so if our torch script fails the first place to look is ``CNNFluilds/torch/lib/include.lua``, and make sure you have all the mandatory libraries.

First compile tfluids (this is our custom CUDA & C++ library that implements a large number of the modules used in the paper):

```
cd FluidNet/torch/tfluids
luarocks make tfluids-1-00.rockspec
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

Warning: We use a 12GB card for training (Titan X). The 32 batch size used on the 3D model requires only 4.5GB during training but at startup libcudnn allocates huge temporary tensors during the first FPROP. We have seen OS crashes using cudnn when trying to allocate too much memory (i.e. when the batch size is too large). This is a cudnn / driver bug.

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

The entire simulation loop is not optimized; we do not consider it fast enough for real-time applications (and we do not claim this in the paper). This is largely because we have not ported our advection code to the GPU, but also because we have not fully optimized all the components (the ConvNet pressure solve is highly optimized and we quote it's runtime in the paper). The advection code is currently the bottleneck of our system.

However, please keep in mind that the primary contribution of this work is a fast approximation to the pressure solve step. This step is profiled during ``fluid_net_train.lua`` startup. The entire 3D simulation step takes about 80ms, which is not terrible, but we do not consider this fast enough to claim it is real-time. **It would be trivial to implement advection on the GPU and reduce this runtime, but we did not have time.**

**BOUNDARY HANDLING**

Our example boundary condition code is very rudimentary. It isn't a limitation of our system, rather that we have not implemented anything more sophisticated. For now, we use a Tensor mask to set pixels occupied (as geometry) or fluid. The grid boundary is assumed to be an empty region. We also have a tensor mask for pressure, velocity and density to set field values constant (this allows us to set in-flow density or velocity regions).

**RENDERING**

We do not have a real-time 3D fluid render. We use an offline render instead. For our 2D "renderer", we simply display the RGB density field to screen and visualize the velocity vectors. It is very rudimentary.

**SIMULATOR**

The only external forces that are supported are vorticity confinement and buoyancy. Viscosity and gravity are not supported (but could be added easily).

Geometry is assumed to be static (i.e. not moving). Again, this is not a limitation of our approach, we just haven't implemented non-static geometry.

This is not necessarily a limitation, but our velocity update does not match the update from mantaflow. There are some cases where manta does not properly calculate the FD of pressure near geometry boundaries.

**UNIT TESTING**

We have unit tests (including FD gradient checks) for all custom torch modules. However we do not have unit tests for everything in tfluids (some parts are tested, but not all).

The two main test scripts we do have are:

```
qlua FluidNet/torch/lib/modules/test_ALL_MODULES.lua
qlua -ltfluids -e "tfluids.test()"
```

You should run these first if you ever get into trouble training or running the model.
