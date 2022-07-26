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
import itertools

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize

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
    tokens = text.split()
    candidates=list()
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
                    candidates.append(w)
    return candidates

def in_expander(text, acronym):
    caps = list(re.findall('(Ch|Ll|[A-Z])', acronym))
    tokens = text.split()
    p_exp = list()
    exp_scores = dict()
    for c in caps:
        exps = list()
        for t in tokens:
            if t[:len(c)]==c:
                exps.append(t)
        p_exp.append(exps)
    print(len(list(itertools.product(*p_exp))))
    for p in list(itertools.product(*p_exp)):
        if len(p)==len(set(p)):
            exp_scores[p] = sum([text.index(w)-text.index(p[0]) for w in p])
    try:
        return ' '.join(min(exp_scores, key=exp_scores.get))
    except:
        return ''
    


total_acr = 0
total_found = 0
total_correct = 0

total_lf = 0
total_lf_found = 0
total_lf_correct = 0

for article in dev:

    text = article["text"]
    acronyms = article["acronyms"]
    long_forms = article["long-forms"]

    acronyms = list(set(coord_to_text(text, acronyms)))
    long_forms = list(set(coord_to_text(text, long_forms)))
    
    article_results = list()
    in_exp_list = list()
    sentences = sent_tokenize(text)
    for s in sentences:
        result = acronym_extractor(s)
        print(result)
    
        for acr in result:
            in_exp = in_expander(s, acr)
            in_exp_list.append(in_exp)

        article_results=[*article_results, *result]


    total_acr += len(acronyms)
    total_found += len(article_results)
    total_correct += len(list(set(acronyms).intersection(article_results)))

    total_lf += len(long_forms)
    total_lf_found += len(in_exp_list)
    total_lf_correct += len(list(set(long_forms).intersection(in_exp_list)))

precision = total_correct / total_found
recall = total_correct / total_acr
F1 = 2 * (precision * recall) / (precision + recall)
print(precision, recall, F1)

lf_precision = total_lf_correct / total_lf_found
lf_recall = total_lf_correct / total_lf

print(lf_precision)
print(lf_recall) 
lf_F1 = 2 * (lf_precision * lf_recall) / (lf_precision + lf_recall)

print(lf_F1)
