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
import re 
import string


from AcronymExtractors.AcronymExtractor import AcronymExtractor


class AcronymExtractor_spanish(AcronymExtractor):

    def acronym_extractor(self, text):
        threshold=0.6
        tokens = text.split()
        acronyms=list()
        for w in tokens:
            w = w.translate(str.maketrans("","", string.punctuation))
            w_t = w
            if len(w_t)>1:
                if 'Ch' in w:
                    w_t = re.sub('Ch', 'C', w)
                elif 'Ll' in w:
                    w_t = re.sub('Ll', 'L', w)
                if bool(any(tilde in w_t for tilde in ['Á', 'É', 'Í', 'Ó', 'Ú'])) == False:
                    if sum(1 for elem in w_t if elem.isupper())/len(w_t)>= threshold:
                        acronyms.append(w)
        return acronyms