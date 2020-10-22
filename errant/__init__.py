import os
from importlib import import_module
import spacy
from errant.annotator import Annotator

# ERRANT version
__version__ = '2.2.2'

# Load an ERRANT Annotator object for a given language
def load(lang, nlp=None):
    # Make sure the language is supported
    supported = {"en", "de"}
    if lang not in supported:
        raise Exception("%s is an unsupported or unknown language" % lang)

    # Load spacy
    nlp = nlp or spacy.load(lang, disable=["ner"])

    # Load language edit merger
    merger = import_module("errant.%s.merger" % lang)

    # Load language edit classifier
    classifier = import_module("errant.%s.classifier" % lang)
    # The English classifier needs spacy
    if lang == "en" or lang == "de":
        classifier.nlp = nlp

    if lang == "de":
        import treetaggerwrapper
        basename = os.path.dirname(os.path.realpath(__file__))
        treetagger = treetaggerwrapper.TreeTagger(TAGLANG="de", TAGDIR=basename + "/resources/tree-tagger-3.2")
    else:
        treetagger = None

    # Return a configured ERRANT annotator
    return Annotator(lang, nlp, merger, classifier, treetagger)