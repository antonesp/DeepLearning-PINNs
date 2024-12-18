Folder structure:

Python scripts in the outer folder can be run to reproduce our results:
	- pinn_all_data_si.py contains our initial testing model, where 	we only attempt to predict Si
	- pinn_regular.py contains a PINN which attempts to estimate all 	paramters in the MVP-model
	- pinn_non_dim.py uses a nondimensionalized version of the PINNS
	- pinn_non_dim_softadapt.py uses nondimensionalized ode's and the 	softadapt library


plots
Contains all data created by running above scripts

SoftAdapt:
Library for softadapt
Borrowed from https://github.com/dr-aheydari/SoftAdapt

functions:
Our helper functions for loading data 

csvs:
Patient.csv and Patient2.csv contain simulated data of the ODE's

DONT-OPEN Our testing ground:
All the weird and failed tests we have tried during the project. You're welcome to look, but we wont promise that anything works.

