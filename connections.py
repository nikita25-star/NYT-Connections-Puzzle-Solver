import numpy as np
import numpy.typing as npt
from util import *

#from graphing import graph_scores, graph_pairwise_similarities

EMBEDDINGS_FILE_PATH = "./openai_embeddings.csv"
PUZZLES_PATH = "./puzzles.json"
GROUP_SIZE = 4
NUM_GROUPS = 4
NUM_WORDS = 16

def find_best_quartet(words: list[str], embeddings: dict[str, npt.NDArray]) -> list[list[str]]:
    '''
    words is the list of 16 words that are jumbled and embeddings is the array that contains 
    the vector points and values for each of the 16 words 
    the list of a list is the array of 4 arrays that is returned sorted,
    creating the best quartet
    '''
    best_score = -float("inf")
    best_quartet = None

    for starting_word in words:
        words_copy = words[:]
        list_of_lists = []

        #source_word = words_copy[0]
        #group = [source_word] #this creates an array group where the source word is the first and only word in in currently
        #words_copy.remove(source_word)

        #source_word = words_copy[0]
        #group.append(source_word)
        #words_copy.remove(source_word)

        for i in range(4): #going through each group of 4, and builds four groups
            group = []
            if i == 0:
                source_word = starting_word #trying each words as the starting point for the first group only 
            else:
                source_word = words_copy[0] 

            group.append(source_word)
            words_copy.remove(source_word)


            while len(group) < 4: #making one group
                group_sum = np.zeros_like(embeddings[group[0]]) #creating the sum variable that holds the total of embeddings of a group
                    #need to initialize sum as a vector because adding other vectors to it to find the centroid_group_embeddings variable
                for word in group:
                    group_sum += embeddings[word]
                centroid_group_embedding = group_sum / len(group)

                greatest_cos_sim = 0
                word_with_greatest_cos_sim = ''
                for word in words_copy: #this checks the cosine similarity between the average of the current group and the next word in the group of 4 words
                    cos_sim = cosine_similarity(centroid_group_embedding, embeddings[word])
            
                    if (cos_sim > greatest_cos_sim): #if the cosine similarity is greater than our pre-established cosine similary, then that is our new greatest similarity and we store that word in a group of 4
                        greatest_cos_sim = cos_sim
                        word_with_greatest_cos_sim = word


                group.append(word_with_greatest_cos_sim) #adding words_copy[i] to group list
                words_copy.remove(word_with_greatest_cos_sim) #removing words_copy[i] from the words list so in later comparison we are not considering this word because we already put it in our group
            list_of_lists.append(group) #appending the group of 4 to the list of lists that have 16 words total and 4 groups total 

        total_score = calculate_total_score(list_of_lists, words, embeddings)
            
        if total_score > best_score:
            best_score = total_score
            best_quartet = list_of_lists
            
        
    return best_quartet



def calculate_group_internal_score(group: list[str], embeddings: dict[str, npt.NDArray], alpha:float=0.75) -> np.floating:
    cosine_similarities = []

    #example group = ['brush', 'shave', 'shower', 'dress']
    #this goes through all the combinations of the 4 words (AB, AC, AD, BC, BD, CD) and appends their internal cosine similaries to an array 
    for i in range(3): # i = 0, 1, 2
        for j in range(i + 1, 4): #j = 1, 2, 3
            vec1 = embeddings[group[i]]
            vec2 = embeddings[group[j]]
            cosine_similarities.append(cosine_similarity(vec1, vec2))
            
    return (alpha * np.mean(cosine_similarities)) + ((1 - alpha) * min(cosine_similarities))

def calculate_group_external_score(group: list[str], other_groups: list[list[str]], embeddings: dict[str, npt.NDArray]) -> np.floating:
    cosine_similarities = []
    
    other_words = []
    for g in other_groups: #traversing through the 3 groups of 4 that are outside the group we have made 
        for w in g: #traversing thorugh words in each group
            other_words.append(w)


    for word_in_group in group: #i = 0, 1, 2, 3
        for word in other_words: #j = 0, 1, 2, 3
            vec1 = embeddings[word_in_group]
            vec2 = embeddings[word]
            cosine_similarities.append(cosine_similarity(vec1, vec2))

    return -np.mean(cosine_similarities) #this value is negative because a high similarity to other groups in this case is bad and a low similarity to other groups is good (because this group is good on its own and should not clash with other groups)
    #adding the negative flips the regular cosine similarity meaning as it means more separation between the words in one group to the external, the higher the external score 

def calculate_word_ambiguity(word: str, words: list[str], embeddings: dict[str, npt.NDArray]) -> float:
    vec1 = embeddings[word]
    greatest = -float("inf") #because cosine_similarity can compute a value for -1.0 to 1.0, initailizing this variable to 0 would not be accurate for finding the greatest similarity value
    for w in words:
        if w == word: #covers the one case where we run into the same word in the list of words of the puzzle
            continue #not breaking full statement, just continuing to the next iteration and not breaking the loop fully 
        vec2 = embeddings[w]
        if (cosine_similarity(vec1, vec2) > greatest):
            greatest = cosine_similarity(vec1, vec2)

    return greatest
    #essentially, this method helps deal with a case where a word is ambiguous or used in multiple contexts
    #a high ambiguity score means that this word can fit into multiple categories as the cosine similarity outputs a higher value and the word is not cleanly separable from all the other words into a singlular group

def calculate_total_score(quartet: list[list[str]], words: list[str], embeddings: dict[str, npt.NDArray],
                          alpha:float=0.8, beta:float=0.2, penalty_weight:float=0.05):
    internal_scores = []
    external_scores = []
    ambiguity_penalties = []
            
    for group in quartet:
        internal_scores.append(calculate_group_internal_score(group, embeddings))

        other_groups = []
        for g in quartet: #scanning the quartet to see if any of them match the current group you are on determined from the top for loop 
            if g is not group: #checking if all groups are or are not the current group I am looping through
                other_groups.append(g)     

        #other_groups array changes every time we loop to the next group in quartet (which happens a total of 4 times) 
        #to calculate the external score for each group in the quartet against every other combination of the other_words array

        external_scores.append(calculate_group_external_score(group, other_groups, embeddings))

        ambiguity_scores = []
        for w in group:
            ambiguity_scores.append(calculate_word_ambiguity(w, words, embeddings)) #array of word ambiguities for each word

        ambiguity_penalties.append(np.mean(ambiguity_scores)) 
        '''contains 4 numbers of the mean of the scores for each group
        #calculating the avg of word ambiguity for each word in a group and appending these 4 averages to the ambiguity_scores array 
        #which then is used in the overall_score calculation and taken mean of then as well to better
        #fit the structure of calculating the mean of the internal and external scores'''

    overall_score = alpha * np.mean(internal_scores) \
                    + beta * np.mean(external_scores) \
                    -penalty_weight * np.mean(ambiguity_penalties)

    '''
    every score is calculated for each group and then the averages of
    those four values are taken to calculate the weighted sum of all 
    three scores
    '''
    
    return overall_score

def main():
    # EXAMPLE USAGE:

    # puzzle #751
    answers = retrieve_puzzles(PUZZLES_PATH)[751]
    todays_words = convert_to_words(answers)

    # hardcoded, this would be:
    # todays_words = ['nick', 'palm', 'pinch', 'pocket', 'brush', 'dress', 'shave', 'shower', 'neat', 'sharp', 'smart', 'tidy', 'birth', 'key', 'mile', 'touch']
    # answers = [['nick', 'palm', 'pinch', 'pocket'], ['brush', 'dress', 'shave', 'shower'], ['neat', 'sharp', 'smart', 'tidy'], ['birth', 'key', 'mile', 'touch']]

    todays_embeddings = get_todays_embeddings(todays_words)
    
    best_q = find_best_quartet(todays_words, todays_embeddings)

    print(find_best_quartet(todays_words, todays_embeddings))


    answer_score = calculate_total_score(best_q, todays_words, todays_embeddings)
    #graph_scores(todays_words, todays_embeddings, "Score Distriution", answer_score)


    #graph_pairwise_similarities(todays_words, todays_embeddings, "Pairwise Similarities")

if __name__ == "__main__":
    main()