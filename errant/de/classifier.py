from pathlib import Path
import Levenshtein
from nltk.stem.snowball import GermanStemmer
from errant.en.lancaster import LancasterStemmer
import spacy
import spacy.symbols as POS

# Load Hunspell word list
def load_word_list(path):
    with open(path) as word_list:
        return set([word.strip() for word in word_list])

# Load Universal Dependency POS Tags map file.
# https://universaldependencies.org/tagset-conversion/en-penn-uposf.html
def load_pos_map(path):
    map_dict = {}
    with open(path) as map_file:
        for line in map_file:
            line = line.strip().split("\t")
            map_dict[line[0]] = line[1].strip()
    return map_dict

# Classifier resources
base_dir = Path(__file__).resolve().parent
# Spacy
nlp = None
# Lancaster Stemmer
stemmer = GermanStemmer#LancasterStemmer()
# GB English word list (inc -ise and -ize)
spell = load_word_list(base_dir/"resources"/"de_DE-large.txt")
# Part of speech map file
pos_map = load_pos_map(base_dir/"resources"/"de-stts_map")
# Contractions
conts = {"'s", "´s"}
# Rare POS tags that make uninformative error categories
rare_tags = {"INTJ", "NUM", "SYM", "X"}
# POS tags with inflectional morphology
inflected_tags = {"ADJ", "ADV", "AUX", "DET", "PRON", "PROPN", "NOUN", "VERB"}
# Fine-grained STTS verb tags
verb_tags = {"VVINF", "VVIZU", "VVPP", "VAINF", "VAPP", "VMINF", "VMPP", "VVFIN", "VVIMP", "VAFIN", "VAIMP", "VMFIN", "VMPP"}
# Some dep labels that map to pos tags.
dep_map = {"ac": "ADP",
	"svp": "ADP",
	"punct": "PUNCT" }

# Input: An Edit object
# Output: The same Edit object with an updated error type
def classify(edit):
    # Nothing to nothing is a detected but not corrected edit
    if not edit.o_toks and not edit.c_toks:
        edit.type = "UNK"
    # Missing
    elif not edit.o_toks and edit.c_toks:
        op = "M:"
        cat = get_one_sided_type(edit.c_toks)
        edit.type = op+cat
    # Unnecessary
    elif edit.o_toks and not edit.c_toks:
        op = "U:"
        cat = get_one_sided_type(edit.o_toks)
        edit.type = op+cat
    # Replacement and special cases
    else:
        # Same to same is a detected but not corrected edit
        if edit.o_str == edit.c_str:
            edit.type = "UNK"
        # Special: Ignore case change at the end of multi token edits
        # E.g. [Doctor -> The doctor], [, since -> . Since]
        # Classify the edit as if the last token wasn't there
        elif edit.o_toks[-1].lower == edit.c_toks[-1].lower and \
                (len(edit.o_toks) > 1 or len(edit.c_toks) > 1):
            # Store a copy of the full orig and cor toks
            all_o_toks = edit.o_toks[:]
            all_c_toks = edit.c_toks[:]
            # Truncate the instance toks for classification
            edit.o_toks = edit.o_toks[:-1]
            edit.c_toks = edit.c_toks[:-1]
            # Classify the truncated edit
            edit = classify(edit)
            # Restore the full orig and cor toks
            edit.o_toks = all_o_toks
            edit.c_toks = all_c_toks
        # Replacement
        else:
            op = "R:"
            cat = get_two_sided_type(edit.o_toks, edit.c_toks)
            edit.type = op+cat
    return edit

# Input: Spacy tokens
# Output: A list of pos and dep tag strings
def get_edit_info(toks):
    pos = []
    dep = []
    for tok in toks:
        pos.append(pos_map[tok.tag_])
        dep.append(tok.dep_)
    return pos, dep

# Input: Spacy tokens
# Output: An error type string based on input tokens from orig or cor
# When one side of the edit is null, we can only use the other side
def get_one_sided_type(toks):
    # Special cases
    if len(toks) == 1:
        # Contraction.
        if toks[0].lower_ in conts:
            return "CONTR"
    # Extract pos tags and parse info from the toks
    pos_list, dep_list = get_edit_info(toks)
    # POS-based tags. Ignores rare, uninformative categories.
    if len(set(pos_list)) == 1 and pos_list[0] not in rare_tags:
        return pos_list[0]
    # More POS-based tags using special dependency labels.
    if len(set(dep_list)) == 1 and dep_list[0] in dep_map.keys():
        return dep_map[dep_list[0]]
    # zu-infinitives
    if set(pos_list) == {"PART", "VERB"}:
        return "VERB"
    # Tricky cases
    else:
        return "OTHER"

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: An error type string based on orig AND cor
def get_two_sided_type(o_toks, c_toks):
    # Extract pos tags and parse info from the toks as lists
    o_pos, o_dep = get_edit_info(o_toks)
    c_pos, c_dep = get_edit_info(c_toks)

    # Orthography; i.e. whitespace and/or case errors.
    if only_orth_change(o_toks, c_toks):
        return "ORTH"
    # Word Order; only matches exact reordering.
    if exact_reordering(o_toks, c_toks):
        return "WO"

    # 1:1 replacements (very common)
    if len(o_toks) == len(c_toks) == 1:
        # 2. SPELLING AND INFLECTION
        # Only check alphabetical strings on the original side
        # Spelling errors take precedence over POS errors; this rule is ordered
        if o_toks[0].text.isalpha():
            # Check a GB English dict for both orig and lower case.
            # E.g. "cat" is in the dict, but "Cat" is not.
            if o_toks[0].text not in spell and \
                    o_toks[0].lower_ not in spell:
                # Check if both sides have a common lemma
                if o_toks[0].lemma == c_toks[0].lemma:
                    # skip to morphology
                    pass
                # Use string similarity to detect true spelling errors.
                else:
                    char_ratio = Levenshtein.ratio(o_toks[0].text, c_toks[0].text)
                    # Ratio > 0.5 means both side share at least half the same chars.
                    # WARNING: THIS IS AN APPROXIMATION.
                    if char_ratio > 0.5:
                        return "SPELL"
                    # If ratio is <= 0.5, the error is more complex e.g. tolk -> say
                    else:
                        # If POS is the same, this takes precedence over spelling.
                        if o_pos == c_pos and \
                                o_pos[0] not in rare_tags:
                            return o_pos[0]
                        # Tricky cases.
                        else:
                            return "OTHER"

        # 3. MORPHOLOGY
        # Only ADJ, ADV, NOUN and VERB can have inflectional changes.
        if o_toks[0].lemma == c_toks[0].lemma and \
                o_pos[0] in inflected_tags and \
                c_pos[0] in inflected_tags:
            # Same POS on both sides
            if o_pos == c_pos:
                if o_pos[0] in inflected_tags:
                    return o_pos[0] + ":FORM"
            # For remaining verb errors, rely on cor_pos
            if c_toks[0].tag_ in verb_tags:
                return "VERB:FORM"
            # Tricky cases that all have the same lemma.
            else:
                return "MORPH"
        # Inflectional morphology.
        if stemmer.stem(o_toks[0].text) == stemmer.stem(c_toks[0].text) and \
                o_pos[0] in inflected_tags and \
                c_pos[0] in inflected_tags:
            return "MORPH"

        # 4. GENERAL
        # POS-based tags. Some of these are context sensitive mispellings.
        if o_pos == c_pos and o_pos[0] not in rare_tags:
            return o_pos[0]
        # Some dep labels map to POS-based tags.
        if o_dep == c_dep and o_dep[0] in dep_map.keys():
            return dep_map[o_dep[0]]
        # Separable verb prefixes vs. prepositions
        if set(o_pos + c_pos) == {"ADP"} or set(o_dep + c_dep) == {"ac", "svp"}:
            return "ADP"
        # Can use dep labels to resolve DET + PRON combinations.
        if set(o_pos+c_pos) == {"DET", "PRON"}:
            # DET cannot be a subject or object.
            if c_dep[0] in {"sb", "oa", "od", "og"}:
                return "PRON"
            # "poss" indicates possessive determiner
            if c_dep[0] == "ag":
                return "DET"
        # Tricky cases.
        else:
            return "OTHER"

    # Multi-token replacements (uncommon)
    # All same POS
    if len(set(o_pos+c_pos)) == 1 and o_pos[0] not in rare_tags:
        return o_pos[0]
    # All same special dep labels.
    if len(set(o_dep+c_dep)) == 1 and \
            o_dep[0] in dep_map.keys():
        return dep_map[o_dep[0]]
    # Infinitives, gerunds, phrasal verbs.
    if set(o_pos+c_pos) == {"PART", "VERB"}:
        # Final verbs with the same lemma are form; e.g. to eat -> eating
        if o_toks[-1].lemma == c_toks[-1].lemma:
            return "VERB:FORM"
        # Remaining edits are often verb; e.g. to eat -> consuming, look at -> see
        else:
            return "VERB"

    # Tricky cases.
    else:
        return "OTHER"

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: Boolean; the difference between orig and cor is only whitespace or case
def only_orth_change(o_toks, c_toks):
    o_join = "".join([o.lower_ for o in o_toks])
    c_join = "".join([c.lower_ for c in c_toks])
    if o_join == c_join:
        return True
    return False

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: Boolean; the tokens are exactly the same but in a different order
def exact_reordering(o_toks, c_toks):
    # Sorting lets us keep duplicates.
    o_set = sorted([o.lower_ for o in o_toks])
    c_set = sorted([c.lower_ for c in c_toks])
    if o_set == c_set:
        return True
    return False
