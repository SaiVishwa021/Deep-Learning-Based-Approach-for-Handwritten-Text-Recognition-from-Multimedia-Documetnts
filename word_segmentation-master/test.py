import nltk
from nltk.corpus import words
from nltk.metrics.distance import edit_distance


# Load your word list from the file
# with open("C:/Users/Pavi/Downloads/word_segmentation/word_segmentation-master/words.txt", "r") as file:
#     word_list = set(file.read().splitlines())
    
# class TrieNode:
#     def __init__(self):
#         self.children = {}
#         self.is_end_of_word = False

# class Trie:
#     def __init__(self):
#         self.root = TrieNode()

#     def insert(self, word):
#         node = self.root
#         for char in word:
#             if char not in node.children:
#                 node.children[char] = TrieNode()
#             node = node.children[char]
#         node.is_end_of_word = True

#     def search(self, prefix):
#         node = self.root
#         for char in prefix:
#             if char not in node.children:
#                 return None  # Prefix not found
#             node = node.children[char]
#         return node

# def correct_spelling(input_text, word_list=word_list):
#     trie = Trie()
#     for word in word_list:
#         trie.insert(word)

#     # Split the input text into words
#     words = input_text.split()

#     corrected_words = []
#     for word in words:
#         prefix = ""
#         corrected_word = ""

#         for char in word:
#             prefix += char
#             node = trie.search(prefix)
#             if node and node.is_end_of_word:
#                 corrected_word = prefix

#         if corrected_word:
#             corrected_words.append(corrected_word)
#         else:
#             # If no valid prefix found, keep the original word
#             corrected_words.append(word)

#     # Join the corrected words back into a sentence
#     corrected_text = ' '.join(corrected_words)

#     return corrected_text

# Read the custom word list from a local file
# with open("C:/Users/Pavi/Downloads/word_segmentation/word_segmentation-master/words.txt", "r") as file:
#     word_list = set(file.read().splitlines())

# def correct_spelling(input_text):
#     # Split the input text into words
#     words = input_text.split()

#     corrected_words = []
#     for word in words:
#         # If the word is in the word list, keep it as is
#         if word in word_list:
#             corrected_words.append(word)
#         else:
#             # Find the closest matching word from the word list
#             closest_word = min(word_list, key=lambda w: edit_distance(word, w))
#             corrected_words.append(closest_word)

#     # Join the corrected words back into a sentence
#     corrected_text = ' '.join(corrected_words)

#     return corrected_text


from spellchecker import SpellChecker

def correct_spelling(input_text):
    # Create a SpellChecker object
    spell = SpellChecker()

    # Split the input text into words
    words = input_text.split()

    # Iterate through the words and correct the spelling
    corrected_words = []
    for word in words:
        corrected_word = spell.correction(word)
        if corrected_word is not None:
            corrected_words.append(corrected_word)
        else:
            # If the word couldn't be corrected, keep the original word
            corrected_words.append(word)

    # Join the corrected words back into a sentence
    corrected_text = ' '.join(corrected_words)

    return corrected_text

