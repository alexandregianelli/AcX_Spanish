'''
5. Ortografía(sigla)

a) Las siglas se escriben hoy sin puntos ni blancos de separación. 
Solo se escribe punto tras las letras que componen las siglas cuando 
van integradas en textos escritos enteramente en mayúsculas: MEMORIA ANUAL DEL C.S.I.C.

b) Las siglas presentan normalmente en mayúscula todas las letras que las componen (OCDE, DNI, ISO) 
y, en ese caso, no llevan nunca tilde; así, CIA (del ingl. Central Intelligence Agency) se escribe sin tilde,
a pesar de pronunciarse [sía, zía], con un hiato que exigiría acentuar gráficamente la i. Las siglas que se 
pronuncian como se escriben, esto es, los acrónimos, se escriben solo con la inicial mayúscula si se trata de 
nombres propios y tienen más de cuatro letras: Unicef, Unesco; o con todas sus letras minúsculas,
si se trata de nombres comunes: uci, ovni, sida. Los acrónimos que se escriben con minúsculas sí deben someterse 
a las reglas de acentuación gráfica (→ tilde2): láser.

c) Si los dígrafos 'ch' y 'll' forman parte de una sigla, va en mayúscula el primer carácter y en minúscula el segundo:
PCCh, sigla de Partido Comunista de China.

d) Se escriben en cursiva las siglas que corresponden a una denominación que debe aparecer en este tipo de letra cuando 
se escribe completa; esto ocurre, por ejemplo, con las siglas de títulos de obras o de publicaciones periódicas: 
DHLE, sigla de Diccionario histórico de la lengua española; RFE, sigla de Revista de Filología Española.

e) Las siglas escritas en mayúsculas nunca deben dividirse con guion de final de línea.
'''
import logging

from nltk.tokenize import word_tokenize

from AcroExpExtractors.AcroExpExtractor import AcroExpExtractorRb


from nltk.tokenize import RegexpTokenizer, sent_tokenize

from AcroExpExtractors.AcroExpExtractor import AcroExpExtractorRb
from abbreviations import schwartz_hearst

from nltk.tokenize import RegexpTokenizer, sent_tokenize

import re
import string
import itertools

log = logging.getLogger(__name__)


class AcroExpExtractor_Spanish(AcroExpExtractorRb):
    def get_all_acronyms(self, doc_text):
        tokens = word_tokenize(doc_text)
        acronyms=list()
        for w in tokens:
            
            w_t = w
            if len(w_t)>1:
                if 'Ch' in w:
                    w_t = re.sub('Ch', 'C', w)
                elif 'Ll' in w:
                    w_t = re.sub('Ll', 'L', w)
                if bool(any(tilde in w_t for tilde in ['Á', 'É', 'Í', 'Ó', 'Ú'])) == False:
                    if w_t.isupper():
                        acronyms.append(w)
        return acronyms


    def get_all_acronym_expansion(self, doc_text):

        abbrev_map = dict()
        omit = 0
        written = 0
        sentences = sent_tokenize(doc_text)
        for s in sentences:
            acronyms = self.get_all_acronyms(s)
            for acr in acronyms:
                abbrev_map[acr] = self.get_best_expansion(acr, s)

        print(abbrev_map)
        return abbrev_map

    def get_all_acronym_expansion(self, doc_text):

        acronyms = self.get_all_acronyms(doc_text)

        abbrev_map = {acronym: None for acronym in acronyms}
        omit = 0
        written = 0

        sentence_iterator = enumerate(schwartz_hearst.yield_lines_from_doc(doc_text))

        for i, sentence in sentence_iterator:
            try:
                for candidate in schwartz_hearst.best_candidates(sentence):
                    try:
                        definition = schwartz_hearst.get_definition(candidate, sentence)
                    except (ValueError, IndexError) as e:
                        log.debug(
                            "{} Omitting candidate {}. Reason: {}".format(
                                i, candidate, e.args[0]
                            )
                        )
                        if candidate not in abbrev_map:
                            abbrev_map[candidate] = None
                        omit += 1
                    else:
                        try:
                            definition = schwartz_hearst.select_definition(
                                definition, candidate
                            )
                        except (ValueError, IndexError) as e:
                            log.debug(
                                "{} Omitting definition {} for candidate {}. Reason: {}".format(
                                    i, definition, candidate, e.args[0]
                                )
                            )
                        if candidate not in abbrev_map:
                            abbrev_map[candidate] = None
                            omit += 1
                        else:
                            abbrev_map[candidate] = definition
                            written += 1
            except (ValueError, IndexError) as e:
                log.debug(
                    "{} Error processing sentence {}: {}".format(i, sentence, e.args[0])
                )
        log.debug(
            "{} abbreviations detected and kept ({} omitted)".format(written, omit)
        )
        return abbrev_map

    def get_acronym_expansion_pairs(self, text):
        newText = "\n".join(sent_tokenize(text))
        return schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=newText)

    def get_best_expansion(self, acro, text):
        best_long_form = ""

        text = text + " (" + acro + ")"

        acr_exp = self.get_acronym_expansion_pairs(text)

        if acro in acr_exp.keys():
            best_long_form = acr_exp[acro]

        return best_long_form
