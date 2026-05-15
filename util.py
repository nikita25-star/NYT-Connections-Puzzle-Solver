"""
File Name: util.py
This program contains some helpful utility functions for connections.py.
"""

import json
import pickle
import os
import pandas as pd
import numpy as np
import numpy.typing as npt

EMBEDDINGS_FILE_PATH = "./openai_embeddings.csv"
PUZZLES_PATH = "./puzzles.json"
GROUP_SIZE = 4
NUM_GROUPS = 4
NUM_WORDS = 16

# retrieve_puzzles takes in one argument, puzzles_path, and returns a dictionary
# mapping the puzzle's id to its answers
def retrieve_puzzles(puzzles_path: str) -> dict[int, list[list[str]]]:
    with open(puzzles_path, "r") as f:
        data = json.load(f)
        puzzles = {}
        for d in data:
            puzzle_id = int(d["id"])

            answer = []
            for a in d["answers"]:
                answer.append(list(map(str.lower, a["members"])))
            
            puzzles[puzzle_id] = answer

        return puzzles

# convert_to_words takes in one argument, answer, and flattens it (thus
# obtaining the original word list)
def convert_to_words(answer: list[list[str]]) -> list[str]:
    return [word for group in answer for word in group]

# fancy_print takes in one argument, quartet, and prints a more readable
# version of it
def fancy_print(quartet: list[list[str]]) -> None:
    # find the max width of each column
    words = convert_to_words(quartet)
    columns = []
    for i in range(NUM_GROUPS):
        columns.append([words[j] for j in range(i, 16, 4)])
    max_lengths = [max(map(len, word)) for word in columns]
    widths = [max(length, 6) + 2 for length in max_lengths]
    
    border = "+" + "+".join(["-" * width for width in widths]) + "+"

    for group in quartet:
        row = "| " + "| ".join([word + (" " * (widths[i] - len(word) - 1)) for i, word in enumerate(group)]) + "|"
        print(border)
        print(row)
        print(border)

# retrieve_embeddings takes in an argument, path, and parses the csv file in the path
# and returns a dictionary mapping each word to its corresponding embedding
def retrieve_embeddings(path: str) -> dict[str, npt.NDArray]:
    #############################################
    pickled_file_path = path.replace(".csv", ".pkl")
    if os.path.exists(pickled_file_path):
        with open(pickled_file_path, "rb") as f:
            return pickle.load(f)
    #############################################

    df = pd.read_csv(path, usecols=["frequency", "word", "embedding"])

    words = df["word"].tolist()
    
    # woah! what is going on here?
    # 1. `pandas`'s `apply` is actually faster than python's built-in `map` or lambdas,
    #    which is why we're using it, considering that these are vectors with thousands of elements.
    # 2. `json.loads` is a function that turns a "stringified list" into a pure list
    # 3. we're casting each embedding as an np.array type for faster calculations--most importantly,
    #    cosine similarities!
    vectors = df["embedding"].apply(json.loads).apply(np.array).tolist()

    embeddings = dict(zip(words, vectors))

    #############################################
    with open(pickled_file_path, "wb") as f:
        pickle.dump(embeddings, f)
    #############################################

    return embeddings

# get_todays_embeddings accepts one argument, todays_words, and returns the embeddings
# of the words in todays_words
def get_todays_embeddings(todays_words: list[str]) -> dict[str, npt.NDArray]:
    todays_embeddings = {}
    
    embeddings = retrieve_embeddings(EMBEDDINGS_FILE_PATH)
    for word in todays_words:
        todays_embeddings[word] = embeddings[word]

    return todays_embeddings

# cosine_similarity returns the cosine of the angle between two vectors
# u and v
def cosine_similarity(u: npt.NDArray, v: npt.NDArray) -> float:
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))    

# check_word_accuracy answers the following question: given the answer to a puzzle,
# how many words are in the right place compared to the answer?
def check_word_accuracy(quartet, answer):
    correct = 0
    for guessed_group, actual_group in zip(quartet, answer):
        correct += len(set(guessed_group).intersection(set(actual_group)))
    return correct / NUM_WORDS

# check_group_accuracy answers the following question: given the answer to a puzzle,
# how many groups are correct? if check_group_accuracy returns a 1, then the solution is correct
def check_group_accuracy(quartet, answer):
    quartet_sets = [set(group) for group in quartet]
    answer_sets = [set(group) for group in answer]

    correct = 0
    for group in quartet_sets:
        if group in answer_sets:
            correct += 1

    return correct / NUM_GROUPS