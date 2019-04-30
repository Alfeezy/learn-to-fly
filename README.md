# learn-to-fly
Python implementation of QLearning on top of a custom gym environment.
Authors: Miranda Yates, Bastien Gliech

#### Description:
This project uses the [openai/gym](https://github.com/openai/gym) toolkit to build and test a flexible QLearning algorithm optimized for flight control in Python. The enviornment used is custom built to simulate an extremely simple flight environment. As the project continues, we hope to eventually create an increasingly complex flight environment. The final goal is to create an action space that resembles that of an actual RC plane, so that we may use the algorithm in a real world scenario. 

#### To Run:
1. ensure that the following python libraries are installed:

   [openai/gym](https://github.com/openai/gym)  
   [numpy](https://github.com/numpy/numpy)  
   [matplotlib](https://github.com/matplotlib/matplotlib)  
   [pandas](https://github.com/pandas-dev/pandas)

2. In the repository folder, run the following:
```
python3 learn.py
```
