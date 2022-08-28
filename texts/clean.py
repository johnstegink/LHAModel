# class to clean text, parts copied from https://github.com/ninikolov/lha

import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import ToktokTokenizer

import functions


class Cleaner():
    WHITESPACE_REGEX = "[ \t\r\f\n]{1,}"

    REGEX_DICT = {
        "&[#\da-zA-Z]*;": "",  # Remove stuff like &apos;
        "\d": r"#",  # Replace all digits with #
        WHITESPACE_REGEX: r" "  # replace all spaces and tabs with a single space
    }

    def __init__(self, language_code):
        """
        Init the cleaner for the given language code (now nl or en)
        :param languageCode: language code
        """

        self.language_code = language_code.lower()
        self.language = functions.translate_language_code( language_code)

        self.stop = stopwords.words(self.language) + list(string.punctuation)
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = SnowballStemmer(self.language)
        self.printable = set(string.printable)
        self.toktok = ToktokTokenizer().tokenize # fast tokenizer




    def __regex_replace(self, txt, reg, replace):
        """
        Use regular expression to replace text
        :param txt:
        :param reg:
        :param replace:
        :return:
        """
        return re.sub(re.compile(reg), replace, txt)



    def __multi_regex_replace(self, txt, regex_dict=REGEX_DICT):
        """
        Use multipe regular expression to replace text
        :param text:
        :param regex_dict:
        :return:
        """
        for match, replace in regex_dict.items():
            txt = self.__regex_replace(txt, match, replace)
        txt = txt.strip(' \t\n\r')
        return " ".join(txt.split())


    def __digit_clean(self, txt):
        """
        Remove digits from the text and replace them with a "#"
        :param txt:
        :return:
        """
        out = ""
        for ch in txt:
            if ch.isdigit():
                out += "#"
            else:
                out += ch
        return out


    def clean_text(self, txt, lower=False, remove_digits=True, remove_stop=True):
        """
        Clean the text according to the parameters
        :param txt:
        :param return_string:
        :param lower:
        :param remove_digits:
        :param remove_stop:
        :return: cleaned list of words
        """

        # Copy and covert to lowercase if neccesairy
        clean = txt.lower() if lower else txt

#        clean = clean.replace("=", " ").replace("*", " ")

        words = self.toktok(clean)

        # Remove stopwords and digits
        clean_words = []
        for word in words:
            if not (word in self.stop and remove_stop) and not (remove_digits and word.isdigit()):
                clean_words.append( word)

        return clean_words # Create a text again



    def printable_text(self, txt):
        """
        Remove not printalbe chars and all new lines and extra whitespaces
        :return: cleaned string
        """
        clean = self.light_clean(txt)
        clean = "".join([char for char in clean if char in self.printable])

        return " ".join(clean.split())



    def light_clean(self, txt):
        """
        Remove all new lines and extra spaces
        :param txt:
        :return:
        """
        # return multi_regex_clean(txt, {WHITESPACE_REGEX: r" "})
        clean = txt.replace('\n', ' ').replace('\r', ' ').replace("\t", " ")
        clean = clean.strip(' \t\n\r')
        return " ".join(clean.split())



    def clean_stem(self, txt):
        """
        Clean and stem words
        :return:
        """
        out = []
        for w in self.tokenizer.tokenize(txt):
            if w not in self.stop and w != '':
                try:
                    w_stem = self.stemmer.stem(w)
                    out.append(w_stem)

                except IndexError:
                    out.append(w)

        return out