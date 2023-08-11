# AlphaPuck

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Licence](https://img.shields.io/github/license/Jens21/AlphaPuck)

<p align="center">
  <img width="300" height="240" src="game_screenshot.png">
</p>


This work was done as part of the Reinforcement Learning (RL) lecture at the University of Tübingen in the summer semester of 2023.
The goal was to develop three RL agents for a 2D two-player hockey game that can beat two PD-controlled basic opponent players.

We implemented Decoupled Q-Networks (DecQN) [[1]](#1), MuZero [[2]](#2) as well as Double Dueling DQN (DDDQN) [[3-4]](#3).
In this repository, we provide both the source code and the trained network parameters.

There was also a final tournament against all RL agents developed by the other participants in the lecture.
**In this tournament, our MuZero agent reached the first place among 89 participants**. The DecQN agent ranked 10th and the DDDQN agend finished in the middle of the field of participants.


## Installation

Since this repository contains submodules, clone and pull with the additional flag `--recurse-submodules`.
To run the scripts in this repository, **Python 3.10** is needed.
Then, simply create a virtual environment and install the required packages via

```bash
pip install -r requirements.txt
```


## Directory Structure:

This repository is structured as follows:

```
src/
├── evaluation/
│   ├── agents/
│   └── main.py
├── fabrice/
├── jens/
└── christoph/
```

**src/evaluation/agents:** Trained network parameters, agent interface.

**src/evaluation/main.py:** Evaluation script (cf. [Usage](#usage)).

**src/fabrice:** DecQN source code.

**src/jens:** MuZero source code.

**src/christoph:** DDDQN source code.



## Usage

The Python script `src/evaluation/main.py` is used to evaluate RL agents against each other or against the two basic opponents Weak and Strong.
It implements a command line interface that allows quick configuration for evaluations.

In the following, we present the most important arguments:

### **--player-1** (mandatory)

Selects the left (protagonist) player ('MuZero', 'DecQN', 'DDDQN', 'Strong', 'Weak'). 

### **--player-2** (mandatory)

Selects the right (opponent) player ('MuZero', 'DecQN', 'DDDQN', 'Strong', 'Weak'). 

### **--num-episodes**

Number of played games.

### **--rng-seed**

Fixes random number generator seed to produce deterministic results.

### **--disable-rendering**

Disables graphical rendering.

For more details, invoke the script with the flag `-h`/`--help`.


## References

<a id="1">[1]</a> 
Tim Seyde et al. "Solving Continuous Control via Q-learning". 
In: The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. 
URL: https://openreview.net/pdf?id=U5XOGxAgccS.

<a id="2">[2]</a> 
Julian Schrittwieser et al. "Mastering Atari, Go, chess and shogi by planning with a learned model".
In: Nature 588.7839 (Dec. 2020), pp. 604–609.
URL: https://doi.org/10.1038%2Fs41586-020-03051-4

<a id="3">[3]</a> 
Ziyu Wang et al. "Dueling Network Architectures for Deep Reinforcement Learning". 
In: Proceedings of The 33rd International Conference on Machine Learning. New York, USA: PMLR, June 2016, pp. 1995–2003.
URL: https://proceedings.mlr.press/v48/wangf16.html.

<a id="4">[4]</a> 
Hado van Hasselt, Arthur Guez, and David Silver. "Deep Reinforcement Learning with Double Q-
Learning". 
In: Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence. AAAI’16. AAAI Press, 2016, pp. 2094–2100.
URL: https://arxiv.org/abs/1509.06461
