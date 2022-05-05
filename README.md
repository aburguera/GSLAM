# A GraphSLAM Front-end For False Loop Rejection

This repository provides a GraphSLAM front-end aimed at rejecting false loops before they corrupt the graph.

| Without false loop rejection | With false loop rejection |
| ------------------- | --------------- |
| ![Not filtering loops](VIDEOS/NOFILTER.gif) | ![Filtering loops](VIDEOS/FILTER.gif) | 

## Credits and citation

**Science and code**: Antoni Burguera (antoni dot burguera at uib dot es)

**Citation**: If you use this software, please cite the following paper:

- Contact the author.

## Basic usage

Check **test_graphoptimizer.py** to see a usage example. The datasets used in this example (and any other provided example) are not available in this repository.

All files are fully commented. Check the comments in each file to learn about it.

## Datasets

The provided examples make use of **DataSimulator** objects (see **datasimulator.py**) to access datasets. DataSimulator is a convenience class to access UCAMGEN datasets pre-processed using **preprocess_dataset.py**. Check **test_datasimulator.py** to learn about how to use **DataSimulator** objects.

Check the [UCAMGEN](https://github.com/aburguera/UCAMGEN) for information about the UCAMGEN format or to grab the software to create synthetic or semi-synthetic UCAMGEN datasets. Check **preprocess_dataset.py** to learn how to pre-process the dataset into the specific **DataSimulator** format.

Neither UCAMGEN datasets nor DataSimulator objects are required to perform loop rejection. They are provided only to make the examples self-contained.

## Understanding the system

All the files are extensively commented. Check the comments to understand how to use the system and how it works. Additionally, check the paper referenced above. Next, the main components are described.

### GraphOptimizer

The main class is **GraphOptimizer**, which has four main methods:

* **add_odometry**: Adds an odometric estimate into the system.
* **add_loop**: Adds a loop constraint into the loop candidate set.
* **validate**: Validates the candidate set.
* **optimize**: Optimizes the graph.

Check the comments in **graphoptimizer.py** for more information. Overall, using the GraphOptimizer is as simple as:

```
theOptimizer=GraphOptimizer(...)
while theRobot.working():
  Xodo=theRobot.get_odometry()
  theOptimizer.add_odometry(Xodo,...)

  loopDetected,Xloop=theRobot.check_loops()
  if loopDetected:
    theOptimizer.add_loop(Xloop,...)
    theOptimizer.validate()
    theOptimizer.optimize()

```

Note that the GraphOptimizer decides if the optimization can be carried out or not. Thus, validate and optimize can be called for every new loop detected since they don't necessarily cause a graph optimization.

The graph optimization is performed using the excellent package [GraphSLAM](https://github.com/JeffLIrion/python-graphslam) by [JeffLIrion](https://github.com/JeffLIrion). However, all the code to perform the optimization is confined in the **optimize** method. So, it can be easily replaced by any other algorithm.

### DataSimulator

The package also provides the class **DataSimulator** which interfaces with [UCAMGEN](https://github.com/aburguera/UCAMGEN) datasets pre-processed with **preprocess_dataset.py**. The file **test_datasimulator.py** shows how to use it and the file **test_graphoptimizer.py** shows how to use the DataSimulator and the GraphOptimizer together. No datasets are provided, though they can be generated using UCAMGEN.

### Loop detection and motion estimation

To provide a complete software, the repository also contains a Siamese Neural Network for loop detection and a RANSAC-based image matcher for motion estimation and basic loop filtering.

The Neural Network code is in the files **loopmodel.py** and **modelwrapper.py**. Please note that the Neural Network weights are not provided. Use [SNNLOOP](https://github.com/aburguera/SNNLOOP) and [AUTOENCODER](https://github.com/aburguera/AUTOENCODER) to train the network with your own datasets.

The RANSAC motion estimator code is provided in **imagematcher.py** and **motionestimator.py** and it belongs to the repository [NNLOOP](https://github.com/aburguera/NNLOOP).

### Evaluation tools

The file **evaluate_all.py** contains a script showing the general workflow to experimentally evaluate each component. The file **evaluate_graphoptimizer.py** shows a particular case of **evaluate_all.py** and is provided just as an easy to understand example.

### Auxiliary code

There are some files with auxiliary functions:

* **util.py** General purpose utilities.
* **plotutil.py** Plotting utilities.
* **transform2d.py** 2D transformations (composition, inversion, ...) and related tools.

## Requirements

To execute this software, you will need:

* Python 3
* Keras and Tensorflow (for SNNLOOP)
* NumPy
* Matplotlib
* Pickle
* SciKit-Image (for SNNLOOP and DataSimulator)
* [GraphSLAM by JeffLIrion](https://github.com/JeffLIrion/python-graphslam)

## Troubleshooting

The GPU activation/deactivation function (set_gpu) is required only for the Neural Network. That function works on my computer (Ubuntu 20 and CUDA Toolkit 10.1) but may not work on other computers, even if they use Ubuntu 20 and CUDA 10.1. Installing CUDA on Ubuntu 20 and making it useable by Keras+Tensorflow is (at least in April 2021) a true nightmare. Really. I don't even know why it works today and didn't work yesterday. So, if it doesn't work do not blame me. **Just remove the set_gpu call, restart the Python console if necessary, and use your own or execute the Neural Network on CPU.**

## Disclaimer

The code is provided as it is. It may work in your computer, it may not work. It may even crash it or, eventually, create a time paradox, the result of which could cause a chain reaction that would unravel the very fabric of the space-time continuum and destroy the entire universe.
