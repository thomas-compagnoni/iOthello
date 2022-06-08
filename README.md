# [Othello-AI](https://github.com/ThomasMind/Othello-AI)

Welcome to Othello-AI, a project which includes basic functions to play Othello, a GUI to help the user-experience and, above all, the implementation of Machine Learning for an AI-Bot. Ready to beat it?

The game is simple to understand, however [here's a brief summary of the rules](https://www.worldothello.org/about/about-othello/othello-rules/official-rules/english).

## An overview
<p>That's it, seems to work mh? Tests show that the AI wins 97% of the time, draws 2% and loses 1% vs a random bot .<br>
In this example the white player is the random bot, the black one is the trained bot.

<p>&nbsp;</p>

<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/cfbfca2f6ce9cc5c99139076cf01f92c3a594b67/figs/match_example.gif" alt="alt text" width="300" height="300">
  
<p>&nbsp;</p>





How does the algorithm works?
-----

I'll follow a top-down approach to explain what's behind, in 3 steps:

1. Building the score function
2. Training the models
3. Optimizing basic functions

### Building the score function

This had been the most difficult task of the whole project, how does the bot choose the best move?
I do not hide the fact that I made dozens of attempts, everyone with bad results, until surprisingly I found the one which worked, moreover well above expectations. 

The functions is composed by two elements:
1. Monte Carlo tree search
2. Machine Learning Regressor

When it's our turn, the algorithm computes all the possible board in a 2 level depth of the **Montecarlo tree**.

#### A visual example

- Suppose the AI is the white player, and we are at the third move in the game. The situation will be like this:

<p>&nbsp;</p>
<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/cfbfca2f6ce9cc5c99139076cf01f92c3a594b67/figs/Move0.png" alt="alt text" width="200" height="200">
<p>&nbsp;</p>

- We need to choose between 5 possible moves. If we compute every move we will have a the first level of the Montecarlo. But we need to go deeper.
- How will be the board if we play move 1?

<p>&nbsp;</p>
<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/cfbfca2f6ce9cc5c99139076cf01f92c3a594b67/figs/Move1.png" alt="alt text" width="200" height="200">
<p>&nbsp;</p>

- The opponent has 5 more moves. Eventually we will have a total of 21 possible boards in the second level of the tree. For example, this is the board if the opponent plays move 3.

<p>&nbsp;</p>
<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/cfbfca2f6ce9cc5c99139076cf01f92c3a594b67/figs/Move2.png" alt="alt text" width="200" height="200">
<p>&nbsp;</p>

#### Now how do we give a score to the board?

- Behind the user-interface, the board is a numpy matrix 6x6 with: 
  - 0 where empty
  - 1 for player one
  - -1 for player two

<p>&nbsp;</p>
<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/cfbfca2f6ce9cc5c99139076cf01f92c3a594b67/figs/Move2_matrix.png" alt="alt text" width="150" height="150">
<p>&nbsp;</p>


