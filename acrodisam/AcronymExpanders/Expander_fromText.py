from AcronymExpanders.AcronymExpander import AcronymExpander
import re
from AcronymExpanders import AcronymExpanderEnum
from string_constants import max_confidence, min_confidence


class Expander_fromText(AcronymExpander):

    def __init__(self, expander_type=AcronymExpanderEnum.fromText):
        AcronymExpander.__init__(self, expander_type)
        self.confidence = max_confidence

    def transform(self, X):
        return [item.getText() for item in X]

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test, acronym):
        expansionsForAllInputs = []
        confidencesForAllInputs = []
        
        for test_input in X_test:
            expansion = self._expandInText(test_input, acronym)

            confidence = min_confidence
            if expansion:
                confidence =self.confidence
                
            expansionsForAllInputs.append(expansion)
            confidencesForAllInputs.append(confidence)

        return expansionsForAllInputs, confidencesForAllInputs

    def _expandInText(self, text, acronym):
        """
        expand acronym from text
        returns:
        acronym expansion or None (if expansion not found in text)
        """
        patterns = self.definition_patterns(acronym)

        for pattern in patterns:
            pattern_results = re.findall(pattern, text)
            if(pattern_results):
                return pattern_results[0]

        return None

    def definition_patterns(self, acronym):
        def_pattern1, def_pattern2 = r'', r''
        between_chars1 = r'\w{3,}[-\s](?:\w{2,5}[-\s]){0,1}'
        between_chars2 = r'\w+[-\s]{0,1}(?:\w{2,5}[-\s]{0,1}){0,1}'
        for i, c in enumerate(acronym):
            c = "[" + c + c.lower() + "]"
            if i == 0:
                def_pattern1 += r'\b' + c + between_chars1
                def_pattern2 += r'\b' + c + between_chars2
            elif i < len(acronym) - 1:
                # acronym letter, chars, periods, space
                def_pattern1 += c + between_chars1
                def_pattern2 += c + between_chars2
            else:
                def_pattern1 += c + r'\w+\b'
                def_pattern2 += c + r'\w+\b'
        acronym = r'' + acronym + r'\b'
        patterns = []
        for def_pattern in [def_pattern1, def_pattern2]:
            patterns = patterns + [def_pattern + r'(?=\sor\s{0,1}(?:the\s){0,1}(?:a\s){0,1}' + acronym + r')',
                                   def_pattern +
                                   r'(?=["(\s,]{2,}(?:or\s){0,1}(?:the\s){0,1}["]{0,1}' +
                                   acronym + r')',
                                   r'(?<=' + acronym + r'\s\W)' + def_pattern]
        # log the patterns
        #logger.debug("Acronym: %s, Patterns:", acronym)
        # for pattern in patterns:
        #    logger.debug(pattern)

        patterns = [re.compile(pattern) for pattern in patterns]
        return patterns
