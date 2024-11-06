# Guess Who Strategy Optimizer
## Overview
For this particular game of Guess Who there are 35 unique characters, meaning that the optimal decision tree would be log base 2 of 35 which is about 5.13 questions to reach the goal if every question is exactly a 50/50 split. This project intends to find the optimal decision tree that has an average depth as close to 5.13 as possible in order to optimize the game.

This project applies data science techniques to optimize the strategy for the classic board game "Guess Who?". Using decision tree algorithms, it determines the most efficient sequence of questions to identify characters in the minimum number of guesses.
Project Highlights

Utilized scikit-learn's DecisionTreeClassifier to model the game decision-making process
Implemented custom depth calculation functions to evaluate tree efficiency
Optimized random state parameters to find the most balanced decision tree
Visualized decision paths and feature importances for strategic analysis

## Technical Details

Language: Python
Key Libraries:

pandas
numpy
scikit-learn
matplotlib



## Methodology

Data Preparation:

Loaded character features from CSV file
Preprocessed features for decision tree compatibility


Optimization Process:

Implemented custom average leaf depth calculation
Evaluated multiple random states (0-100,000) to find optimal tree structure
Selected random_state=166467 based on minimum average leaf depth

Analysis Features:

Tree depth visualization
Feature importance ranking
Node count and leaf analysis
Path reachability verification



## Key Findings

Successfully created a decision tree that ensures all characters are uniquely identifiable
Generated visualizations to understand the most important character features
Optimized question sequence to minimize the number of guesses needed

## Usage

Clone the repository
Ensure you have the required Python packages installed
Place your Guess Who character dataset in the specified location
Run GuessWho.py to generate the optimized decision tree

