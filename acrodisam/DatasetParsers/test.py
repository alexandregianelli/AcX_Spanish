from process_tokens_and_bio_tags import (
    _match_capital_initials,
    _ratio_match_initials,
    _match_initials_capital_acro
)


acronyms = ['PCCh'] 
long_forms = ['Partido Comunista Chino']

def match_acronyms(acronyms, long_forms):
    for acronym in acronyms:
        for lf in long_forms:
            #print(_ratio_match_initials(acronym, lf.split(), 0.5))
            #print(_match_capital_initials(acronym, lf.split()))
            print(_match_initials_capital_acro(acronym, lf.split()))   
            if _match_capital_initials(acronym, lf.split())== True:
               print(acronym, lf)
               
            
                    


match_acronyms(acronyms, long_forms)