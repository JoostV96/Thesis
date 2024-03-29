# Thesis
Code that is used in the Master Thesis: "Zeroth-order optimization for machine learning approaches"

It consists of two files: 
- ARS.py: the code for the Augmented Random Search.
- EM.py:  the code for the Electromagnetism-like Mechanism.



**Prerequisites:** 

Download Python 3.x -> https://www.python.org/downloads/

Install anaconda -> https://www.anaconda.com/distribution/

Open the Anaconda Navigator and install your preferred IDE (e.g. Spyder)

Use the Anaconda Promt and run the following command:
```
pip install gym
```
Next we need to install the PyBullet environment. 

For this we first need the microsoft visual C++ buildtools.

After installing this, run the following commands in the Anaconda prompt:
```
pip install pybullet
conda install -c menpo ffmpeg
```

To run the Hopper task set `self.env_name = "HopperBulletEnv-v0"` and adjust the parameters accordingly.

To run the Inverted Double Pendulum task set `self.env_name = "InvertedDoublePendulumBulletEnv-v0"` and adjust the parameters accordingly.

NOTE: the code for ARS is retrieved from:
https://github.com/sourcecode369/Augmented-Random-Search-/blob/master/Augmented%20Random%20Search/ars.py
