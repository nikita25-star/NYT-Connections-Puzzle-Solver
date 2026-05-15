import random
import matplotlib.pyplot as plt
import seaborn as sns
from util import *
from connections import *

NUM_BINS = 20
NUM_SAMPLES = 1_000
LINE_WIDTH = 2
TITLE_FONT_SIZE = 10
HEATMAP_WIDTH = 11
HEATMAP_HEIGHT = 10

# generate_random_quartet takes in a list of words from a puzzle and generates a random quartet out of those
# 16 words; if from_word_bank is set to be True, then this function will randomly sample 16 words from this _huge_
# list of words and generate a random quartet from those 16 words
def generate_random_quartet(words: list[str], from_word_bank:bool=False) -> list[list[str]]:
    if from_word_bank:
        shuffled = random.sample(words, NUM_WORDS)
    else:
        shuffled = words[:]  # make a copy
    random.shuffle(shuffled)
    return [shuffled[i:(i + GROUP_SIZE)] for i in range(0, GROUP_SIZE * NUM_GROUPS, GROUP_SIZE)]

def graph_pairwise_similarities(todays_words: list[str], todays_embeddings: dict[str, np.float64], title: str) -> None:

    size = len(todays_words)
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            vec1 = todays_embeddings[todays_words[i]] #computing vectorizations of todays_words[i] and todays_words[j] from the todays_embeddings array
            vec2 = todays_embeddings[todays_words[j]]
            similarity = cosine_similarity(vec1, vec2)
            matrix[i][j] = similarity #adds the 16^2 values of similarities between every word combination comparison to add to the heatmap 

    plt.figure(figsize=(10, 8))

    sns.heatmap(matrix, xticklabels=todays_words, yticklabels=todays_words, cmap="coolwarm", square=True, annot=True, fmt=".2f") #creates a heatmap using the pairwise cosine similarity matrix, with the words as labels on the x and y axes, and uses the "coolwarm" color scheme to visually represent the similarities between words
    plt.xticks(rotation=45, ha="right") #tilting the x labels to be read vertically and algins them so they don't overlap 
    plt.title(title, fontsize=TITLE_FONT_SIZE, color='black') #setting the title of the graph from the title parameter given in the function 
    plt.show()

def graph_scores(todays_words: list[str], todays_embeddings: dict[str, np.float64], title: str, 
                 answer_score:np.floating|None=None, from_word_bank:bool=False) -> None:
    random_scores = []
    for i in range(1000):
        random_quartet = generate_random_quartet(todays_words, from_word_bank)
        score = calculate_total_score(random_quartet, todays_words, todays_embeddings)
        random_scores.append(score)

    wyna_quartet = [['nick', 'palm', 'pinch', 'pocket'], ['brush', 'dress', 'shave', 'shower'], ['neat', 'sharp', 'smart', 'tidy'], ['birth', 'key', 'mile', 'touch']]
    wyna_score = calculate_total_score(wyna_quartet, todays_words, todays_embeddings)
    
    plt.figure(figsize=(10, 6)) #creating the histogram figure
    sns.histplot(random_scores, bins=NUM_BINS, kde=True, color='skyblue', edgecolor='black') #actually plots the random scores generated into 30 distinct invervals and draws a kernal density estimate curve on top of the bars 

    plt.axvline(wyna_score, color='purple', linestyle='dashed', linewidth=LINE_WIDTH, label="Wyna's Answer")#plots a vertical line at the score of wyna's answer to show where it is in relation to the distribution of random scores and solver's score
    
    #need to check if the answer_score is none before we graph the line for the solver 
    if answer_score is not None:
        plt.axvline(answer_score, color='blue', linestyle='dashed', linewidth=LINE_WIDTH, label="Solver's Solution")#plots a vertical line at the score of the solver's answer to show where it is in relation to the distribution of random scores and wyna's score



    plt.axvline(np.mean(random_scores), color='green', linestyle='dashed', linewidth=LINE_WIDTH, label="Mean")#creates a line for the mean of the random scores generated 


    plt.xlabel('Coherence Score', fontsize=TITLE_FONT_SIZE, color='black')
    plt.ylabel('Count', fontsize=TITLE_FONT_SIZE, color='black')
    plt.title(title, fontsize=TITLE_FONT_SIZE, color='black')

    plt.legend()
    plt.show()
    

if __name__ == "__main__":

    answers = retrieve_puzzles(PUZZLES_PATH)[751]
    todays_words = convert_to_words(answers)

    # hardcoded, this would be:
    # todays_words = ['nick', 'palm', 'pinch', 'pocket', 'brush', 'dress', 'shave', 'shower', 'neat', 'sharp', 'smart', 'tidy', 'birth', 'key', 'mile', 'touch']
    # answers = [['nick', 'palm', 'pinch', 'pocket'], ['brush', 'dress', 'shave', 'shower'], ['neat', 'sharp', 'smart', 'tidy'], ['birth', 'key', 'mile', 'touch']]

    todays_embeddings = get_todays_embeddings(todays_words)
    best_q = find_best_quartet(todays_words, todays_embeddings)
    answer_score = calculate_total_score(best_q, todays_words, todays_embeddings)

    graph_scores(todays_words, todays_embeddings, "Score Distriution", answer_score)

    graph_pairwise_similarities(todays_words, todays_embeddings, "Pairwise Similarities")
