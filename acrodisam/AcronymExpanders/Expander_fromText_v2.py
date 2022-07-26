import re

from AcronymExpanders import AcronymExpanderEnum
from string_constants import max_confidence, min_confidence
from AcronymExpanders.Expander_fromText import Expander_fromText



class Expander_fromText_v2(Expander_fromText):
    """
    Changes over Expander_fromText:
    multiline regex
    regex will now look for more than one line breaks between expansions (changing between_chars1)
        needs a join on split up expansion: " ".join([word for word in pattern_result[0].split()])
    changing regex between_chars to support acronyms like: Integrated Definition (IDEF)
    """

    def __init__(self):
        Expander_fromText.__init__(self, AcronymExpanderEnum.fromText_v2)

        # less confidence than fromText
        self.confidence = (
            (max_confidence - min_confidence) * 0.9) + min_confidence

    def _expandInText(self, text, acronym):
        patterns = self.definition_patterns(acronym)

        for pattern in patterns:
            pattern_results = re.findall(pattern, text)
            if(pattern_results and pattern_results[0] != acronym):
                return pattern_results[0]

        return None

    def definition_patterns(self, acronym):
        def_pattern = r''
        between_chars1 = r'\w*[-.\s]*'  # (?:\w{2,5}[-\s]){0,1}'
        # between_chars2 = r'\w*[-\s]*(?:\w{2,5}[-\s]{0,1}){0,1}'
        after_chars = r"\w*"
        between_def_and_acronym = r"[-,\"'\(\s\w]{0,10}"
        for i, c in enumerate(acronym):
            c = "[" + c + "]"
            if i == 0:
                def_pattern += r'\b' + c + between_chars1
            elif i < len(acronym) - 1:
                # acronym letter, chars, periods, space
                def_pattern += c + between_chars1
            else:
                def_pattern += c + after_chars
        patterns = []
        patterns = patterns + ["(" + def_pattern + ")" + between_def_and_acronym + acronym,
                               acronym + between_def_and_acronym + "(" + def_pattern + ")"]
        # log the patterns
        #logger.debug("Acronym: %s, Patterns:", acronym)
        # for pattern in patterns:
        #    logger.debug(pattern)

        patterns = [
            re.compile(pattern, re.MULTILINE | re.IGNORECASE) for pattern in patterns]
        return patterns
