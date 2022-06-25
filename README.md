# [iOthello](https://github.com/ThomasMind/iOthello)

Welcome to iOthello, a project which leverages the power of machine learning to create an AI-Bot for Othello and it includes a GUI to play against.
*Ready to beat it?*

The game is simple to understand, however if you don't know the rules [here's a brief summary](https://www.worldothello.org/about/about-othello/othello-rules/official-rules/english).

## Getting Started

Run on the command line:

```bash
pip install iothello
```

Or if you use the Anaconda distribution:

```bash
conda install iothello
```

Now the package is ready for use. 
From your python environment run:

```python
import iothello

iothello.play()
```

## An overview
In the example below the white player is a random bot, the black one is the trained bot.<br>
Tests show that the AI wins 97% of the time, draws 2% and loses 1% vs a random bot.


<p>&nbsp;</p>

<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/cfbfca2f6ce9cc5c99139076cf01f92c3a594b67/figs/match_example.gif" alt="alt text" width="300" height="300">
  
<p>&nbsp;</p>


How does the algorithm works?
-----

Click to go directly to the paragraph:

1. [Building the score function](#1-building-the-score-function)
2. [How to find the best move](#2-how-to-find-the-best-move)
3. [Training the models with machine learning](#3-training-the-models)

## 1. Building the score function

This had been the most difficult task of the whole project, how does the bot choose the best move? <br>
I made dozens of attempts, everyone with bad results, until surprisingly I found the one which worked, moreover well above expectations. 

The function is composed by two elements:
1. A Monte Carlo tree search
2. A Regressor scoring each board

### 1.1 Monte Carlo tree search

When it's our turn, the algorithm computes all the possible board in a **Montecarlo tree** of depth 2.

#### A visual example

- Suppose the AI is the white player, and we are at the third move in the game. The situation can be like this:

<p>&nbsp;</p>
<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/27f5747e7704b5d8d0c2ef01a72de60aced2f8a9/figs/move0.png" alt="alt text" width="200" height="200">
<p>&nbsp;</p>

- We need to choose between 5 possible moves. They represent the first level of our tree. But we need to go deeper.
- How will be the board if we play move 5?

<p>&nbsp;</p>
<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/27f5747e7704b5d8d0c2ef01a72de60aced2f8a9/figs/move1.png" alt="alt text" width="200" height="200">
<p>&nbsp;</p>

- The opponent has 4 moves. 
- In this example, all the possible moves in response to our first move will give us a total of 21 possible boards, this is the second level of the montecarlo tree. For example, this is the board if the opponent plays move 1.

<p>&nbsp;</p>
<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/27f5747e7704b5d8d0c2ef01a72de60aced2f8a9/figs/move2.png" alt="alt text" width="200" height="200">
<p>&nbsp;</p>

### 1.2 The Regressor
#### Now how do we give a score to each board?

The objective is to reduce the board to a single number, which depends on the number of pieces a player has and their position on the board. As in chess a rook and a queen are usually stronger than a knight and a bishop, in othello pieces on the edges and corners are stronger.
Reading other projects on Github, I found out that the scores are often given discretionarily, by setting manually some rules. This method can't be optimized, how can a human determine the perfect parameters?
What I did was different, I let a machine learning model to decide whether a move was convenient or not. <br>

- Machine learning needs numbers, so, behind the user-interface, the board is a numpy matrix 6x6:<br><p align="center"><br>**0** where empty | **1** for player one | **-1** for player two

<p>&nbsp;</p>
<p align="center">
  <img src="https://github.com/ThomasMind/Othello-AI/blob/27f5747e7704b5d8d0c2ef01a72de60aced2f8a9/figs/move2_matrix.png" alt="alt text" width="150" height="150">
<p>&nbsp;</p>

- The first approach was to simply sum the pieces on the board, so if player **1** has more pieces the score is positive, otherwise it will be negative. But this implies that every cell has the same importance, we need to include also the information given by the position of the cell.

**Finally we can write our formula:**
> The score is a weighted sum of the board

$$ \Huge score = \sum_{i=1}^{6}\sum_{j=1}^{6} w_{m,i,j}*c_{i,j} $$

- **The parameters**: *w* is the weight, *c* is the value of a singular cell. 
- **The subscripts**: *m* is the move number (weights are dinamic through the match), *i* is the line, *j* the column.

> In the image above: 
> a simple sum gives -1.0, while my model gives -1.5. It is saying that the black position is stronger than white's.
  
### 2. How to find the best move
Now we can assign a score to a given board, but how do we confront them?<br> 
We need to apply the score function on every board computed in the Montecarlo tree. For every board in the first level we'll have a list of scores for the ones in the second layer.<br>
The image explains the process, I used the same example has before to make it clearer. Every circle represent a board, the color depends on the player which has the turn. From the starting position (the root node), we developed our possible moves and each subsequent move.<br>
What is new here is the last line **MIN**. This is the minimum score for each subset which can be intrepreted as the worst case-scenario if we choose that path. *We want to minimize this risk*.
> **The best move is the one with the highest *MIN* value.**

<p align="center">
<img src="https://github.com/ThomasMind/Othello-AI/blob/12f70cab51547ccf68f00a579ac4c4199fd5b9d1/figs/scores.png" alt="alt text" width="550" height="350"> 


Basically it is a **Minimax** function. This is a concept well known in game theory, I'll leave you the wikipedia [page](https://en.wikipedia.org/wiki/Minimax).
  
> #### Some facts:
>  - We can apply the rule to player two by inverting the steps (first MAX than MIN), more the score is negative more the player has an advantage.
>  - When there is a move which makes the opponent skips his turn, the AI will likely choose it.
>  - We can run the score function with weights = 1 for every move and every cell, in this case the probabilities of winning fall to 75%.
 
 
## 3. Training the models

- In order to train a model we need data.
- In research.py I implemented the function "random_simulations" which simulates **n** random matches between two random bots.
- For every move I save the board as a flattened array of dimension 1x36 which will represent our X<br>The final score of the match, which is a number, is our y
- The function returns X, y:
  - X is an 3D matrix of dimension (32, **n**, 36)
  - y is a vector with n elements
  
___An example___:
  
- Suppose we have done 4 simulations and we want to see what are the possible states of the board at move 10. <br> With the command X[10, : , : ] the result will be:

$$
  \small X = 
  \left[\begin{array}{c}
  1&0&-1&0&0&0&1&1&-1&0&0&0&0&1&1&-1&0&0&-1&-1&-1&1&-1&0&0&0&0&0&1&-1&0&0&0&0&0&0\\
  0&0&0&1&-1&0&0&-1&0&-1&1&0&1&1&1&1&1&0&0&-1&1&-1&0&0&0&0&0&-1&0&0&0&0&0&-1&0&0\\
  0&0&0&0&0&0&0&1&0&1&-1&0&0&0&1&1&0&0&0&0&1&1&-1&-1&0&1&-1&0&-1&0&0&1&-1&0&-1&0\\
  0&1&0&0&0&0&0&-1&1&-1&1&1&0&0&1&1&1&0&0&0&1&1&1&-1&0&0&1&0&0&0&0&0&1&0&0&0
  \end{array}\right]
$$
  
- Each row represents a single simulation, while each columns represents a different cell of the board.
  
- Let's see the y:

$$
  y = 
  \left[\begin{array}{c}
  18\\
  18\\
  -16\\
  10
  \end{array}\right]
$$
  
- The input for our machine learning model are ready. Each columns is a features, each row is a sample.
- For each move we are building a different model, that is for every row in the 1st axis of X.


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
- The graph on the left is the same of the graph above but with cluster grouping, while the graph on the left is representing what are the cells corresponding to each cluster.
<p>&nbsp;</p>
<p align="center">
<img src="https://github.com/ThomasMind/Othello-AI/blob/fc49a4d314ca5ed621a86f5731dc464e51a62f52/figs/weights_clusters.png" alt="alt text" width="450" height="330"><img src="https://github.com/ThomasMind/Othello-AI/blob/293d50d2b1db5db85207080897ec143ee2141044/figs/board_clustered.png" alt="alt text" width="377" height="330">
  
- The results are astonishing, they are intuitive and they respect the classical theory of the game.
- The corners are the most powerful cells, they can't be taken.
- The cell near them have negative weights at the start because they allow the opponent to take the corners.
- The edges have some tactical power, they are difficult to be taken too.
- Notice that the weights converge to 1, the score of the last move is simply the sum of the board, this will be more clear when we'll talk about how the models were trained
  
### Model selection
  
- The model was not selected as a normal machine learning would do, so by dividing the dataset in train and test and then selecting the model with the highest score and highest generalization out-of-sample.
- I decided to test it directly in some matches against a random bot and choose the model with the highest chances of winning. In research.py this was done by the function *multiprocessing_test_vs_random_bot()* which uses the advantages of parallel computing to play multiple matches at the same time and so reducing the time of execution.
- We chose the ridge regression because it reduces the problems of multicollinearity by keeping low values of the weights. Indeed, as the board is often symmetric after a sequence of moves the cells are naturally dependent on each other.
  
  
 
