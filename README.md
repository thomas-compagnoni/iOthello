# [Othello-AI](https://github.com/ThomasMind/Othello-AI)

Welcome to Othello-AI, a project which includes basic functions to play Othello, a GUI to help the user-experience and, above all, the implementation of Machine Learning for an AI-Bot. *Ready to beat it?*

The game is simple to understand, however [here's a brief summary of the rules](https://www.worldothello.org/about/about-othello/othello-rules/official-rules/english).

## An overview
<p>That's it, seems to work, isn't it? Tests show that the AI wins 97% of the time, draws 2% and loses 1% vs a random bot .<br>
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

## 1. Building the score function

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
  <img src="https://github.com/ThomasMind/Othello-AI/blob/27f5747e7704b5d8d0c2ef01a72de60aced2f8a9/figs/move0.png" alt="alt text" width="200" height="200">
<p>&nbsp;</p>

- We need to choose between 5 possible moves. If we compute every move we will have a the first level of the Montecarlo. But we need to go deeper.
- How will be the board if we play move 5?

<p>&nbsp;</p>
<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/27f5747e7704b5d8d0c2ef01a72de60aced2f8a9/figs/move1.png" alt="alt text" width="200" height="200">
<p>&nbsp;</p>

- The opponent has 4 more moves. Eventually we will have a total of 21 possible boards in the second level of the tree. For example, this is the board if the opponent plays move 1.

<p>&nbsp;</p>
<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/27f5747e7704b5d8d0c2ef01a72de60aced2f8a9/figs/move2.png" alt="alt text" width="200" height="200">
<p>&nbsp;</p>

#### Now how do we give a score to the board?

- Behind the user-interface, the board is a numpy matrix 6x6 with: **0** where empty | **1** for player one | **-1** for player two

<p>&nbsp;</p>
<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/27f5747e7704b5d8d0c2ef01a72de60aced2f8a9/figs/move2_matrix.png" alt="alt text" width="150" height="150">
<p>&nbsp;</p>

> **The score is simply a weighted score of the board**

$$ \Huge score = \sum_{i=1}^{6}\sum_{j=1}^{6} w_{m,i,j}*c_{i,j} $$

- The parameters: **w** is the weight, **c** is the value of a singular cell. 
- The subscripts: *m* is the move number (weights are dinamic through the match), *i* is the line, *j* the column.

- The weights are determined by a Ridge Regression with alpha=1, we'll see later how to train it.

### Weights

- Let's explore the weights. Here is the graph of how they evolve through the time, every line represent a cell. The legend shows the coordinate of the matrix.
- We can immediately spot three important facts:
  1. Some weights move similarly
  2. Their value change through the game, some even move from being negative to being positive.
  3. Some have values near zero.
<p align="center">
<img src="https://github.com/ThomasMind/Othello-AI/blob/293d50d2b1db5db85207080897ec143ee2141044/figs/weights.png" alt="alt text" width="600" height="350"> 

<p>&nbsp;</p>

- Now we want to build clusters which come out naturally, they are 6. We make the average in-cluster and call them with capital letters.
<p>&nbsp;</p>
<p align="center">
<img src="https://github.com/ThomasMind/Othello-AI/blob/293d50d2b1db5db85207080897ec143ee2141044/figs/weights_clusters.png" alt="alt text" width="480" height="350"> <img src="https://github.com/ThomasMind/Othello-AI/blob/293d50d2b1db5db85207080897ec143ee2141044/figs/board_clustered.png" alt="alt text" width="450" height="350">
  
- The results are astonishing, they are intuitive and they respect the classical theory of the game.
- The corners are the most powerful cells, they can't be taken.
- The cell near them have negative weights at the start because they allow the opponent to take the corners.
- The edges have some tactical power, they are difficult to be taken too.
- Note that the weights converge to 1, the score of the last move is simply the sum of the board, this will be more clear when we'll talk about how the models were trained
  
### Final evaluation of the move
  
- Now we need to apply the score function on every board computed in the Montecarlo tree.
- For the example before: score = -1.56397049

<p align="center">
<img src="https://github.com/ThomasMind/Othello-AI/blob/12f70cab51547ccf68f00a579ac4c4199fd5b9d1/figs/scores.png" alt="alt text" width="550" height="350"> 
  
- We now need only two final steps to find the best move, for every subset we find the minimum score, which is shown at the bottom in the picture. This can be intrepreted as the worst case-scenario after the response move of the opponent. We want to minimize that risk.
- The best move is the one with the maximum value, in the example above it is the move 3.
> We can apply the rule also to player two by inverting the steps, before we find the maxima than we choose the move with the minimum value.

I didn't know it before, but it is a **Minimax** function. This is a concept well known in game theory, I'll leave you the wikipedia [page](https://en.wikipedia.org/wiki/Minimax).
  
> **Some facts**:
>  - When there is a move which makes the opponent skips his turn, the AI will likely choose it.
>  - The othello 6x6 has been already "solved", the perfect match exists. Our algorithm respects the first 5 move, then it deviates.
>  - We can run the score function with weights = 1 for every move and every cell (it's the sum of the board), in this case 
>    the probabilities of winning fall to 75%.
 
 
## 2. Training the models
  
