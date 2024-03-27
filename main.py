import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn


def convert_english_text(input_text: str) -> np.ndarray:
    latin_letters = ['F', 'U', 'TH', 'O', 'R', 'C', 'G', 'W', 'H', 'N', 'I', 'J', 'EO', 'P', 'X', 'S', 'T', 'B', 'E',
                     'M', 'L', 'ING', 'OE', 'D', 'A', 'AE', 'Y', 'IA', 'EA']

    e2p = {}
    for i in range(0, 29):
        e2p[latin_letters[i]] = i
    e2p["IO"] = e2p["IA"]
    e2p["K"] = e2p["C"]
    e2p["NG"] = e2p["ING"]
    e2p["Z"] = e2p["S"]
    e2p["Q"] = e2p["C"]
    e2p["V"] = e2p["U"]

    input_text = input_text.upper().replace("QU", "KW")
    input_text = input_text.replace("Q", "K")

    text_as_index = []

    skip = 0

    for index, value in enumerate(input_text):
        if skip:
            skip -= 1
            continue

        elif (index < len(input_text) - 3) and (input_text[index:index + 3] in e2p):
            result = input_text[index:index + 3]
            skip = 2

        elif (index < len(input_text) - 2) and (input_text[index:index + 2] in e2p):
            result = input_text[index:index + 2]
            skip = 1

        else:
            result = input_text[index]

        text_as_index.append(e2p[result])

    return np.asarray(text_as_index)


def gp_sum(input_array: np.ndarray) -> int:
    gp = np.array(
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
         109])
    return sum(gp[input_array])


if __name__ == '__main__':
    nltk.data.path.append('./nltk_data/')
    vowels = ("a", "e", "i", "o", "u")
    all_nouns = [word for synset in wn.all_synsets('n') for word in synset.lemma_names()]
    just_letters = [word for word in all_nouns if re.match("[a-zA-Z]*$", word)]
    unique_words = list(set(just_letters))
    consonant_start = [word for word in unique_words if not word.lower().startswith(vowels)]
    in_indices = list(map(convert_english_text, consonant_start))
    length_8_words = [word for word, indices in zip(consonant_start, in_indices) if len(indices) == 8]
    length_8_indices = [indices for indices in in_indices if len(indices) == 8]

    sorting_indices = np.argsort(length_8_words)
    words = np.array(length_8_words)[sorting_indices]
    indices = np.array(length_8_indices)[sorting_indices]
    gp_sums = np.fromiter(map(gp_sum, indices), dtype=int)
    gp_sums_read_header = gp_sums + 97
    plaintext_minus_ciphertext = np.mod(indices - np.array([19, 21, 23, 27, 2, 14, 10, 19]), 29)
    ciphertext_minus_plaintext = np.mod(np.array([19, 21, 23, 27, 2, 14, 10, 19]) - indices, 29)
    plaintext_plus_ciphertext = np.mod(indices + np.array([19, 21, 23, 27, 2, 14, 10, 19]), 29)
    df = pd.DataFrame({"Word": words.tolist(),
                       "Indices": indices.tolist(),
                       "GP sum": gp_sums.tolist(),
                       "GP sum red header": gp_sums_read_header.tolist(),
                       "Plaintext minus ciphertext": plaintext_minus_ciphertext.tolist(),
                       "Plaintext plus ciphertext": plaintext_plus_ciphertext.tolist(),
                       "Ciphertext_minus_plaintext": ciphertext_minus_plaintext.tolist()
                       })
    df.to_csv("cribbing_list_full.csv", header=True, index=False, sep=';')

    df_1 = df[0:3500]
    df_2 = df[3501::]
    df_1.to_csv("cribbing_list_part1.tsv", header=True, index=False, sep="\t")
    df_2.to_csv("cribbing_list_part2.tsv", header=True, index=False, sep="\t")
