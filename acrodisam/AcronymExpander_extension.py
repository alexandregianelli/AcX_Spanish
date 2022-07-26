"""Modified acronym_expander for an optimized use of the browser extension

@author : jpereira, mprieur
"""

import collections
import sys

from AcronymExpanders.AcronymExpander import PredictiveExpander
from acronym_expander import AcronymExpander
from helper import AcronymExpansion, TestInstance
from string_constants import min_confidence

from Logger import logging


logger = logging.getLogger(__name__)

class AcronymExpander_Extension(AcronymExpander):
    """
    Class to process (text/pdf files) and expand the acronyms in it
    """

    def __init__(self, text_extractor,
                 textPreProcess,
                 acroExpExtractor,
                 expanders,
                 articleDB,
                 acronymDB,
                 expansionChooser=None,
                 linksFollower=None):
        """
        Args:
        text_extractor (TextExtractors.TextExtractor)
        acronym_extractor (AcronymExtractors.AcronymExtractor)
        expanders (list): list of AcronymExpanders
        articleDB: DataCreators.ArticleDB
        acronymDB: DataCreators.AcronymDB
        expansionChooser (lambda): function which takes in a list of AcronymExpansion
                                    and returns a list of AcronymExpansion with only chosen
                                    expansions. If not supplied, _chooseAmongstExpansions
                                    will be used
        """
        super().__init__(text_extractor,
                         textPreProcess,
                         acroExpExtractor=acroExpExtractor,
                         expanders=expanders,
                         articleDB=articleDB,
                         acronymDB=acronymDB,
                         expansionChooser=expansionChooser,
                         links_follower=linksFollower)

    def processText(self, text, testArticleID=None, text_with_links=None, base_url=None):
        """
        takes text and returns the expanded acronyms in it
        Args:
        file_text: text to expand acronyms in. Use extractText API to get text from file
        Returns:
        dict(acronym (unicode): list(helper.AcronymExpansion))
        """
        expansions_per_acronym = {}
        expansion_dict = self.acroExpExtractor.get_all_acronym_expansion(text=text)
        acronyms = [acronym for acronym, expansion in expansion_dict.items() if not expansion]
        if self.links_follower and text_with_links:
            expansion_dict = self.links_follower(text_with_links, expansion_dict, base_url)
        pre_processed_text = self.textPreProcess(text, expansion_dict)
        test_instance = TestInstance(article_id=testArticleID,
                                     article_text=pre_processed_text, acronym="N/A")
        if len(self.acronymExpanders) == 1:
            x_test = self.acronymExpanders[0].transform([test_instance])
        for acronym in acronyms:
            expansions = []
            x_train, y_train, label_to_expansion = self._getChoices(acronym)
            logger.debug("X_train: %s %s", str(len(x_train)), str(sys.getsizeof(x_train)))
            logger.debug("y_train: %s %s", str(len(y_train)), str(sys.getsizeof(y_train)))
            options = [l[1] for l in label_to_expansion.items()]
            for expander in self.acronymExpanders:
                # check if this is a suitable problem for predictive expanders
                if isinstance(expander, PredictiveExpander):
                    if len(x_train) == 0:
                        # no point using prediction, no training data
                        # move to next expander
                        continue
                    if len(label_to_expansion) == 1:
                        # no point using prediction, all same class
                        # predict as the only present class
                        expansion = AcronymExpansion(
                            expansion=label_to_expansion[0],
                            expander=expander.getType,
                            confidence=min_confidence,
                            options=options)
                        expansions.append(expansion)
                        continue
                logger.debug("To transform X_train")
                x_transformed = expander.transform(x_train)
                logger.debug("To fit")
                expander.fit(x_transformed, y_train)
                if len(self.acronymExpanders) > 1:
                    x_test = expander.transform([test_instance])
                results, confidences = expander.predict(x_test, acronym)
                result = results[0]
                confidence = confidences[0]
                if isinstance(expander, PredictiveExpander):
                    # always predicts, no need to check for None
                    expansions.append(AcronymExpansion(expansion=label_to_expansion[result],
                                                       expander=expander.getType(),
                                                       confidence=confidence,
                                                       options=options))
                else:
                    # expansion from non-predictive may sometimes be None
                    if result:
                        expansions.append(
                            AcronymExpansion(expansion=result,
                                             expander=expander.getType(),
                                             confidence=confidence,
                                             options=options))
            expansions_per_acronym[acronym] = self.expansionChooser(expansions)
        external_expansions = {acronym:acronymexpansion[0].expansion\
            for acronym, acronymexpansion in expansions_per_acronym.items()\
                 if len(acronymexpansion) > 0}
        expansion_dict.update(external_expansions)
        return collections.OrderedDict(sorted(expansion_dict.items()))
