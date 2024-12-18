Folder structure:

Python scripts in the outer folder can be run to reproduce our results:
	- pinn_all_data_si.py contains our initial testing model, where 	we only attempt to predict Si
	- pinn_regular.py contains a PINN which attempts to estimate all 	paramters in the MVP-model
	- pinn_non_dim.py uses a nondimensionalized version of the PINNS
	- pinn_non_dim_softadapt.py uses nondimensionalized ode's and the 	softadapt library


plots
Contains all data created by running above scripts

SoftAdapt:
fun

