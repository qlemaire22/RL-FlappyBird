### Reinforcement Learning - Flappy Bird

Project of Deep Learning in Data science (KTH DD2424):

* Quentin Lemaire
* Quentin Vecchio
* Kevin Yeramian

### Installation on Ubuntu

1. `sudo apt-get update`
2. `sudo apt-get install python python3-pip` (You can also use `python2.7`)
3. `sudo apt-get install git`
4. `git clone https://github.com/ntasfi/PyGame-Learning-Environment`
5. `cd PyGame-Learning-Environment`
6. `pip install -e . && cd ..`
7. `pip install future tensorflow opencv-python` (You can choose between `tensorflow` and `tensorflow-gpu`)
8. `pip install pygame`

### Run training

`python training.py`

### Run agent

`python play.py`

By default, the script `play.py` use the old network model. If you want to use your own model (trained with the training.py script) you have to edit the script `play.py` to use the Network class and not the NetworkOld class.

### References

This work is based on the code of Yen-Chen Lin (https://github.com/yenchenlin/DeepLearningFlappyBird/blob/master/deep_q_network.py).
