# focus may be more into preventing the parser in finding expansions at all cost
# and allow it to return no expansions so the acronym is discarded.

import re
import os
import sys
sys.path.append('..')
import AcronymExpanders
from AcronymExpanders import AcronymExpanderEnum
from AcronymExpanders.Expander_fromText import Expander_fromText
from AcronymExtractors import AcronymExtractor_v4
#from AcronymExtractor_v4 import get_acronyms
from string_constants import max_confidence, min_confidence
from string_constants import folder_data, FILE_ENGLISH_WORDS

class Expander_fromText_v3(Expander_fromText):
    """
    Leah Expander 20/6/2018
    """
    english_words = set(word.strip().lower() for word in open(FILE_ENGLISH_WORDS))

    def __init__(self):
        Expander_fromText.__init__(self, AcronymExpanderEnum.fromText_v2)
        # less confidence than fromText
      #  self.confidence = (
      #      (max_confidence - min_confidence) * 0.9) + min_confidence

    def text_snippet(self, acronym, text, index, num):
        text_length = len(text)
        acronym_index = index #text.index(acronym)
        if acronym_index + num >= text_length:
            sub_text =  text[acronym_index-num : -1]
            return sub_text
        if acronym_index - num <= 0:
            sub_text = text[0 : acronym_index+num]
            return sub_text
        if acronym_index == 0:
            sub_text = text[acronym_index:(acronym_index+num)]
            return sub_text
        if acronym_index == -1:
            sub_text = text[acronym_index-num:]
            return sub_text
        else: #if acronym_index + 150 < text_length and acronym_index - 150 > text_length and acronym_index != 0 and acronym_index != -1
            sub_text = text[acronym_index-num:acronym_index] + text[acronym_index:acronym_index+num]
            return sub_text
        return sub_text

    def acronym_indices(self, acronym, article): #looks for every index of the acronym in a document
        index_list = []
        for match in re.finditer(acronym, article):
            index_list.append(match.start())
        return index_list

    def definition_patterns(self, acronym):  # Create definition regex patterns from acronym
        def_pattern1, def_pattern2, def_pattern3 = r'', r'', r''
        between_chars1 = r'\w{2,}[-\s](?:\w{2,5}[-\s]){0,1}'
        between_chars2 = r'\w+[-\s]{0,1}(?:\w{2,5}[-\s]{0,1}){0,1}'
        for i, c in enumerate(acronym):
            c = "[" + c + c.lower() + "]"
            if i == 0:
                def_pattern1 += r'\b' + c + between_chars1
                def_pattern2 += r'\b' + c + between_chars2
            elif i < len(acronym) - 1:
                def_pattern1 += c + between_chars1  # acronym letter, chars, periods, space
                def_pattern2 += c + between_chars2
            else:
                def_pattern1 += c + r'\w+\b'
                def_pattern2 += c + r'\w+\b'
        acronym = r'' + acronym + r'\b'
        patterns = []
        for def_pattern in [def_pattern1, def_pattern2]:
            patterns = patterns + [def_pattern + r'(?=\sor\s{0,1}(?:the\s){0,1}(?:a\s){0,1}' + acronym + r')',
                               def_pattern + r'(?=["(\s,]{2,}(?:or\s){0,1}(?:the\s){0,1}["]{0,1}' + acronym + r')',
                               def_pattern + r'(?=[-\s]{2,}(?:of\s){0,1}'+ acronym + r')', r'(?<=' + acronym + r'\s\W)' + def_pattern]
        patterns = [re.compile(pattern) for pattern in patterns]
        return patterns

    def text_expand(self, acronym, text, patterns):  # Search original text for acronyms
        for pattern in patterns:
            pattern_result = re.findall(pattern, text)
            if pattern_result:
                return pattern_result[0]
        return None



    def acronym_regular_expression(self, acronym, text): #another function to expand acronyms
        new_rx = r'\b'
        if 's' == acronym[-1]:
            acronym = acronym[:-1]
        acronym.lower()
        for i in range(len(acronym)):
            if i == (len(acronym) - 1):
                new_rx += r'[' + acronym[i].lower() + acronym[i].upper() + '_' + ']' + '\w*[\s,.]{0,1}' #+ '(?:of\s){0,1}(?:the\s){0,1}(?:a\s){0,1}(?:and\s){0,1}(?:,\s){0,1}(?:for\s){0,1}(?:.\s){0,1}'
            else:
                new_rx += r'[' + acronym[i].lower() + acronym[i].upper() + '_' + ']' + '\w*[\s,.]{0,1}\s' + '(?:in\s){0,1}(?:of\s){0,1}(?:the\s){0,1}(?:a\s){0,1}(?:and\s){0,1}(?:,\s){0,1}(?:for\s){0,1}(?:.\s){0,1}'
        new_rx = new_rx + r'\b'
        pattern = re.compile(new_rx)
        result = re.findall(pattern, text)
        if result:
            for i in range(len(result)):
                if result[i].isdigit():
                    return None
                else:
                    return result[0]
        return None

    def acronym_uppercase_finder(self, acronym, text): #finds acronyms that have uppercase letters
        new_rx = r'\b'
        new_result = ""
        if 's' == acronym[-1]:
            acronym = acronym[:-1]
        for i in range(len(acronym)):
            new_rx += r'[' + acronym[i] + '_' + ']' + "\w*\s*" + "[\s',.-]{0,1}" + "\w*\s*"
        new_rx = new_rx + r'\b'
        pattern = re.compile(new_rx)
        result = re.findall(pattern, text)
        for i in range(len(result)):
            if result[i].find(acronym):
                if result[i].isdigit():
                    return None
                else:
                    new_result = result[i]
        return new_result

    def acronym_brackets_letter_correspondence(self, acronym, text): #finds expansions within brackets
        temp = ''
        result = re.findall('\((.*?)\)', text)
        for i in range(len(result)): #hierarchy to search within the brackets
            temp = self.acronym_regular_expression(acronym, text)
        if not temp:
            return result
        return temp


    def _expandInText(self, acronym, text):
        num = 150
        acronym_expansion_list = []
        patterns = self.definition_patterns(acronym)

        index_list = self.acronym_indices(acronym, text)
        final_expansion = ''

        for index in index_list:
            smaller_text = self.text_snippet(acronym, text, index, num)
            definition = self.text_expand(acronym, smaller_text, patterns)
            brackent_correspondence = self.acronym_brackets_letter_correspondence(acronym, smaller_text)
            expansion = self.acronym_regular_expression(acronym, smaller_text)
            finder = self.acronym_uppercase_finder(acronym, smaller_text)

            if definition:
                 if definition not in acronym_expansion_list:
                     final_expansion = definition
                     acronym_expansion_list.append(definition)

            elif brackent_correspondence:
                 if brackent_correspondence not in acronym_expansion_list:
                     final_expansion = brackent_correspondence
                     acronym_expansion_list.append(brackent_correspondence)

            elif expansion:
                 if expansion not in acronym_expansion_list:
                     final_expansion = expansion
                     acronym_expansion_list.append(expansion)

            elif finder:
                 if finder not in acronym_expansion_list:
                     final_expansion = finder
                     acronym_expansion_list.append(finder)


        if not acronym_expansion_list:
            return ""
        else:
            return acronym_expansion_list
