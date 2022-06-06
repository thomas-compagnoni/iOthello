# [Othello-AI](https://github.com/ThomasMind/Othello-AI)

Welcome to Othello-AI, a project which includes basic functions to play Othello, a GUI to help the user-experience and, above all, the implementation of Machine Learning for an AI-Bot. Ready to beat it?

The game is simple to understand, however [here's a brief summary of the rules](https://www.worldothello.org/about/about-othello/othello-rules/official-rules/english).

## An overview
<p>That's it, seems to work mh? Tests show that the AI-Bot vs a random bot wins 97% of the time, draws 2% and loses 1%.<br>
In this example the white player is a random bot, the black one is the trained bot.

<p>&nbsp;</p>

<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/main/match_example.gif" alt="alt text" width="300" height="300">
  
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

When it's our turn, the algorithm computes all the possible board in a 2 level depth of the Montecarlo tree.




