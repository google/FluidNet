-- Copyright 2016 Google Inc, NYU.
-- 
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
-- 
--     http://www.apache.org/licenses/LICENSE-2.0
-- 
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- conf table with default parameters.
--
-- Everything here can be modified through the command line (see README.md for
-- more info). Note: newModel is actually moved out of the conf table at
-- model creation and becomes the "mconf" table. This table is saved to disk
-- on every epoch so that simulations can be paused and restarted.

function torch.defaultConf()
  -- Please keep this table in alphabetical order.
  local conf = {
    batchSize = 16,  -- Definitely depends heavily on model and dataset
    dataDir = '../data/datasets/',  -- Where the unprocessed data is stored.
    dataset = 'output_current_3d_model_sphere',  -- Default: 2D with obstacles.
    -- We might want to disable evaluation on test-set while training to save
    -- time.
    evaluateDuringTraining = true,
    gpu = 1,  -- Cuda GPU to use
    ignoreFrames = 0,  -- Ignore the first 'n' frames of each run
    -- lrEpochMults: pairs of {epoch, multiplier}. We will
    -- apply the specified multiplier to the learning rate at each epoch.
    lrEpochMults = {{epoch = math.huge, mult = 0.25}},
    loadModel = false,  -- set true when resuming training or evaluating
    --loadVoxelModel: used in fluids_net_3d_sim.lua only. Options:  'none |
    -- arc | bunny'
    loadVoxelModel = 'none',
    maxEpochs = 5000,  -- Maximum number of epochs
    maxSamplesPerEpoch = math.huge,  -- For debugging.
    modelDir = '../data/models/',
    modelFilename = 'conv_model',  -- Output model file name
    newModel = {
      addBatchNorm = false,
      -- addPressureSkip: If true add a pressure skip connection.
      addPressureSkip = false,
      -- advectionMethod: options are 'euler', 'maccormack'
      advectionMethod = 'maccormack',
      banksJoinStage = 3,  -- Join BEFORE this stage.
      banksAggregateMethod = 'concat',  -- options are 'concat' and 'add'
      banksNum = 1,  -- Number of parallel resolution banks (1 == disable).
      banksSplitStage = 1,  -- Split BEFORE this stage.
      -- banksType: Controls the network structure. Either 'mres' (for multiple
      -- resolution banks, i.e. Gaussian pyramid followed by Conv, followed
      -- by upscale) or 'dilate' (for dilated convolutions without any
      -- explicit upscale or downscale. 'dilate' cannot share weights.
      banksType = 'mres',
      banksWeightShare = false,
      batchNormAffine = true,  -- ignored if addBatchNorm == false.
      batchNormEps = 1e-4,  -- ignored if addBatchNorm == false.
      batchNormMom = 0.1,  -- ignored if addBatchNorm == false.
      -- buoyancyScale: Buoyancy force scale. Set to 0 to disable. 
      buoyancyScale = 0,
      -- dt: default simulation timestep. We will check this against manta
      -- data when training.
      dt = 0.1,
      -- gradNormThreshold: if the L2 norm of the gradient vector goes above
      -- the threshold then we will re-scale it to the threshold value.
      -- This is vitally important in removing outliers.
      gradNormThreshold = 1,
      -- gravityScale: Gravity force scale. Set to 0 to disable.
      gravityScale = 0,
      -- inputChannels: Specify which inputs will be sent to the projection
      -- network.
      inputChannels = {
        div = true,
        flags = true,
        pDiv = true,
        UDiv = false,
      },
      lossFunc = 'fluid',  -- Only fluid is supported for now.
      lossFuncBorderWeight = 4,  -- 1 == disabled.
      lossFuncBorderWidth = 3,  -- We linearly ramp from 1 to weight.
      lossPLambda = 0,
      lossULambda = 0,
      lossDivLambda = 1,
      -- longTermDivLambda: Set to 0 to disable (or set longTermDivNumSteps to
      -- nil).
      longTermDivLambda = 1.0,
      -- longTermDivNumSteps: We want to measure what the divergence is after
      -- a set number of steps for each training and test sample. Set table
      -- to nil to disable, (or set longTermDivLambda to 0).
      longTermDivNumSteps = {4, 16},
      -- longTermDivProbability is the probability that longTermDivNumSteps[1] 
      -- will be taken, otherwise longTermDivNumSteps[2] will be taken with
      -- probability of 1 - longTermDivProbability.
      longTermDivProbability = 0.9,
      -- optimizationMethod: available options: 'sgd', 'adam', 'adagrad',
      -- 'lbfgs' (requires full batch not mini batches)
      modelType = 'default',  -- Choices are 'default', 'yang', 'tog'
      nonlinType = 'relu',  -- Choices are: 'relu', 'relu6', 'sigmoid'.
      normalizeInput = true,  -- If true, normalize by max(std(chan), thresh)
      normalizeInputChan = 'UDiv',  -- Which input channel to calculate std.
      normalizeInputFunc = 'std',  -- Choices are: 'std' or 'norm' (l2).
      normalizeInputThreshold = 0.00001,  -- Don't normalize input noise.
      optimizationMethod = 'adam',
      optimState = {
        bestPerf = math.huge,
        learningRate = 0.0025,
        weightDecay = 0,  -- L2 regularization parameter
        momentum = 0.9,
        dampening = 0,
        learningRateDecay = 0,
        nesterov = false,
        epsilon = 0.0001,  -- epsilon value for ADAM optimizer.
        beta1 = 0.9,  -- beta1 value for ADAM optimizer.
        beta2 = 0.999,  -- beta2 value for ADAM optimizer.
      },
      poolType = 'avg', -- avg or max.
      -- simMethod: At inference time (or even during training
      -- 'tfluids.simulate' calls for the future frame divergence inputs) we
      -- can choose to use a different simulator (other than convnet). This
      -- allows comparisons with our 'jacobi' or 'pcg' baseline methods. Use
      -- 'convnet' to simluate using our method.
      simMethod = 'convnet',
      timeScaleSigma = 1,  -- Amplitude of time scale perturb during training.
      -- trainBuoyancyProb - We will add buoyancy to the training with
      -- this probability Note that this is ONLY applied to the future frame
      -- divergence term.
      trainBuoyancyProb = 0,
      -- trainBuoyancyScale -- This is the buoyancy scale to use when adding
      -- buoyancy. We will apply the scale in a random cardinal direction
      -- (i.e. up, down, left, right, forward, backward).
      trainBuoyancyScale = 2,
      trainGravityProb = 0,
      trainGravityScale = 2,
      -- trainTargetSource -- We might not want to use Manta's pressure and
      -- velocity targets as ground truth. This is particularly true if we want
      -- to add pressure or velocity terms for the future frames (since we
      -- wont have Manta GT for these). Instead, you can use 'jacobi' and 'pcg'
      -- as the solver sources. This will slow down training substantially
      -- especially if you use Jacobi. In practice, the Jacobi solver will be
      -- inaccurate and may not converge, so you should probably keep the
      -- lossPLambda and lossULambda terms small. Otherwise, to use Manta's
      -- output set to 'manta'.
      trainTargetSource = 'manta',
      trainVorticityConfinementProb = 0,
      trainVorticityConfinementAmp = 2,
      -- vorticityConfinementAmp: The vorticity confinement scale value.
      -- Set to 0 to disable vorticity confinement.
      vorticityConfinementAmp = 0.0,
    },
    numDataThreads = 8,  -- To amortize the cost of data loading / processing.
    profile = true,  -- If true then profile the model at startup.
    resumeTraining = false,
    statsSimMethod = '',  -- Empty means use mconf value.
    train = true,  -- perform training (otherwise just evaluate)
  }
  return conf
end

