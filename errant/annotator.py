import re

from errant.alignment import Alignment
from errant.edit import Edit
from spacy.tokens import Doc

# Main ERRANT Annotator class
class Annotator:

    # Input 1: A string language id: e.g. "en"
    # Input 2: A spacy processing object for the language
    # Input 3: A merging module for the language
    # Input 4: A classifier module for the language
    def __init__(self, lang, nlp=None, merger=None, classifier=None, treetagger=None):
        self.lang = lang
        self.nlp = nlp
        self.merger = merger
        self.classifier = classifier
        self.treetagger = treetagger

    # Input 1: A text string
    # Input 2: A flag for word tokenisation
    # Output: The input string parsed by spacy
    def parse(self, input_text, tokenise=False):
        if tokenise:
            text = self.nlp(input_text)
        else:
            text = Doc(self.nlp.vocab, input_text.split())
            self.nlp.tagger(text)
            self.nlp.parser(text)

        if self.treetagger:
            import treetaggerwrapper
            tokens = []
            if tokenise:
                tokens = [token.text for token in text]
            else:
                tokens = input_text.split()
            tags = treetaggerwrapper.make_tags(self.treetagger.tag_text("\n".join(tokens) + "\n", tagonly=True))
            if len(tokens) == len(tags):
                for i in range(0, len(tags)):
                    # use treetagger lemmas
                    if isinstance(tags[i], treetaggerwrapper.Tag):
                        text[i].lemma_ = tags[i].lemma

                    # if spacy provides an empty tag (as for —),
                    # use treetagger tag, with an exception for German punctuation
                    if text[i].tag_ == "":
                        text[i].tag_ = tags[i].pos
                        if re.match(r'^\p{P}+$', text[i].text):
                            if self.lang == "de":
                                text[i].tag_ = "$("

        # check (again) for empty tags (as for —),
        # check for punctuation, otherwise use XX
        for tok in text:
            if tok.tag_ == "":
                tok.tag_ = "XX"
                if re.match(r'^\p{P}+$', tok.text):
                    if self.lang == "en":
                        tok.tag_ = ":"
                    elif self.lang == "de":
                        tok.tag_ = "$("
        return text

    # Input 1: An original text string parsed by spacy
    # Input 2: A corrected text string parsed by spacy
    # Input 3: A flag for standard Levenshtein alignment
    # Output: An Alignment object
    def align(self, orig, cor, lev=False):
        return Alignment(orig, cor, lev)

    # Input 1: An Alignment object
    # Input 2: A flag for merging strategy
    # Output: A list of Edit objects
    def merge(self, alignment, merging="rules"):
        # rules: Rule-based merging
        if merging == "rules":
            edits = self.merger.get_rule_edits(alignment)
        # all-split: Don't merge anything
        elif merging == "all-split":
            edits = alignment.get_all_split_edits()
        # all-merge: Merge all adjacent non-match ops
        elif merging == "all-merge":
            edits = alignment.get_all_merge_edits()
        # all-equal: Merge all edits of the same operation type
        elif merging == "all-equal":
            edits = alignment.get_all_equal_edits()
        # Unknown
        else:
            raise Exception("Unknown merging strategy. Choose from: "
                "rules, all-split, all-merge, all-equal.")
        return edits

    # Input: An Edit object
    # Output: The same Edit object with an updated error type
    def classify(self, edit):
        return self.classifier.classify(edit)

    # Input 1: An original text string parsed by spacy
    # Input 2: A corrected text string parsed by spacy
    # Input 3: A flag for standard Levenshtein alignment
    # Input 4: A flag for merging strategy
    # Output: A list of automatically extracted, typed Edit objects
    def annotate(self, orig, cor, lev=False, merging="rules"):
        alignment = self.align(orig, cor, lev)
        edits = self.merge(alignment, merging)
        for edit in edits:
            edit = self.classify(edit)
        return edits

    # Input 1: An original text string parsed by spacy
    # Input 2: A corrected text string parsed by spacy
    # Input 3: A token span edit list; [o_start, o_end, c_start, c_end, (cat)]
    # Input 4: A flag for gold edit minimisation; e.g. [a b -> a c] = [b -> c]
    # Input 5: A flag to preserve the old error category (i.e. turn off classifier)
    # Output: An Edit object
    def import_edit(self, orig, cor, edit, min=True, old_cat=False):
        # Undefined error type
        if len(edit) == 4:
            edit = Edit(orig, cor, edit)
        # Existing error type
        elif len(edit) == 5:
            edit = Edit(orig, cor, edit[:4], edit[4])
        # Unknown edit format
        else:
            raise Exception("Edit not of the form: "
                "[o_start, o_end, c_start, c_end, (cat)]")
        # Minimise edit
        if min: 
            edit = edit.minimise()
        # Classify edit
        if not old_cat: 
            edit = self.classify(edit)
        return edit
