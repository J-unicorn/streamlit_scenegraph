import spacy
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words
# object and subject constants
OBJECT_DEPS = {"dobj", "dative", "attr", "oprd"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
# tags that define wether the word is wh-
WH_WORDS = {"WP", "WP$", "WRB"}


# extract the subject, object and verb from the input
def extract_svo(doc):
    sub = []
    at = []
    ve = []
    for token in doc:
      
        # is this a verb?
        if token.pos_ == "VERB":
            ve.append(token.text)
        # is this the object?
        if token.dep_ in OBJECT_DEPS or token.head.dep_ in OBJECT_DEPS:
            at.append(token.text)
        # is this the subject?
        if token.dep_ in SUBJECT_DEPS or token.head.dep_ in SUBJECT_DEPS:
            sub.append(token.text)
    return " ".join(sub).strip().lower(), " ".join(ve).strip().lower(), " ".join(at).strip().lower()

# wether the doc is a question, as well as the wh-word if any
def is_question(doc):
    # is the first token a verb?
    if len(doc) > 0 and doc[0].pos_ == "VERB":
        return True, ""
    # go over all words
    for token in doc:
        # is it a wh- word?
        if token.tag_ in WH_WORDS:
            return  token.text.lower()
    return False, ""

def token_result(text):
    s,v,o=extract_svo(nlp(text))
    s='<QUESTION>' if s == is_question(nlp(text)) else '<SUBJECT>'+ s.replace('the ','')
    v='<QUESTION>' if v == is_question(nlp(text)) else '<PREDICATE>'+v.replace('ing','')  
    o='<QUESTION>' if o == is_question(nlp(text)) else  '<OBJECT>'+ o 
    return ' '.join([s,v,o])