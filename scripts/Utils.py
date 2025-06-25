import pandas as pd
import demoji
import regex as re

from amseg.amharicNormalizer import AmharicNormalizer as normalizer
from amseg.amharicSegmenter import AmharicSegmenter


class DataUtils:
    def remove_emoji(self, text):
        try:
            no_emoji = demoji.replace(text, repl = "")
        except:
            no_emoji = text

        return no_emoji


    def tokenizer(self, message):
    
        punct = ['።', '፤', '፡', '!', '?', '፥', '፦', '፧', '(', ')', ',', '.', '-']
        sent_punct = []
        word_punct = []
        undesired_words = ['@classy', 'ብርands', 'ብርandseller', '@sami_twa', '@kingsmarque']
        segmenter = AmharicSegmenter(sent_punct,word_punct)
        words = segmenter.amharic_tokenizer(message)

        return [word for word in words if word not in punct and not re.match(r'[a-zA-Z]', word) and word not in undesired_words]

    def label_conll_format(self, word_list):
        labeled_data = []
        i = 0
        while i < len(word_list):
            word = word_list[i]

            if word == "ዋጋ":
                # Label 'ዋጋ' as B-Price and next two words as I-Price
                labeled_data.append((word, "B-Price"))
                if i + 1 < len(word_list):
                    labeled_data.append((word_list[i + 1], "I-Price")) # I-Price for the actual price like 1000
                if i + 2 < len(word_list):
                    labeled_data.append((word_list[i + 2], "I-Price")) # this will be for ብር part
                i += 3

            elif word == "አድራሻ":
                # Label 'አድራሻ' as O, then next two words as location
                labeled_data.append((word, "O"))
                if i + 1 < len(word_list):
                    labeled_data.append((word_list[i + 1], "B-LOC")) # Addis
                if i + 2 < len(word_list):
                    labeled_data.append((word_list[i + 2], "I-LOC")) # Ababa
                if i + 3 < len(word_list):
                    labeled_data.append((word_list[i + 3], "I-LOC")) # hayahulet           
                if i + 4 < len(word_list):
                    labeled_data.append((word_list[i + 4], "I-LOC")) # 22 <- since the sefer hayahulet is a number               
                i += 5

            else:
                # Label all other words as O
                labeled_data.append((word, "O"))
                i += 1

        return labeled_data

        