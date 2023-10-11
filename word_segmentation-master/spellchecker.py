import nltk
from nltk.corpus import words
from nltk.metrics.distance import edit_distance





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

