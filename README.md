# Q-Learning_Maze
A reinforcement learning model Q-learning used in simple maze game.

## Introduction

 - A training model on a simple maze game: blue square is the character; green square is the pickup which can increase, +1 point; then the red square is the trap that decrease, -1 point.
 
 - The model uses Q-learning, including main QValue Network and target QValue Network.
 
 - ~~The final training on test may sometimes gets stucked, please try again before I fix it.~~I have fix it at 2017.12.03, and add another type of object, and a mode that randomly gives number of each kind of objexts.
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/277.gif?raw=true)
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/278.gif?raw=true)
 
*********************

## How to use

 - Choose to train or to test by setting the _toTest_ and if you want to test, set _testSteps_ to set the action steps that character allow to take.
