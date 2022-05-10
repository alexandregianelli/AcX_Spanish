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
import os
import json
import string


data_path = os.getcwd()+'/AcX_Spanish-main/AcX_Spanish-main/data/AcronymExpansion/'
print(data_path)
dev= json.load(open(data_path+'/dev.json', encoding='utf-8'))
train = json.load(open(data_path+'/train.json', encoding='utf-8'))
test = json.load(open(data_path+'/test.json', encoding='utf-8'))

def coord_to_text(text, coords):
    result = list()
    for coord in coords:
        result.append(text[coord[0]:coord[1]].replace('(', '').replace(')', ''))
    return result

def acronym_extractor(text):
    threshold=0.6
    token = text.split()
    candidates=list()
    for w in token:
        w = re.sub('[(,)]', '', w)
        if 'Ch' in w or 'Ll' in w:
            pass
        if sum(1 for elem in w if elem.isupper())/len(w)>= threshold:
            candidates.append(w)
    return candidates
total_acr = 0
total_found = 0
for article in dev:

    text = article["text"]
    acronyms = article["acronyms"]
    long_forms = article["long-forms"]

    acronyms = list(set(coord_to_text(text, acronyms)))
    long_forms = list(set(coord_to_text(text, long_forms)))
    
    result = acronym_extractor(text.translate(str.maketrans('', '', string.punctuation)))
    total_acr += len(acronyms)
    total_found += len(list(set(acronyms).intersection(result)))

score = total_found / total_acr
print(score)
