import argparse
import os
import spacy
import scripts.align_text as align_text
import scripts.toolbox as toolbox

def main(args):
	# Get base working directory.
	basename = os.path.dirname(os.path.realpath(__file__))
	print("Loading resources...")
	spacy_disable = ['ner']
	treetagger = None
	if args.lang == "en":
		from nltk.stem.lancaster import LancasterStemmer
		import scripts.cat_rules as cat_rules
		# Load Tokenizer and other resources
		nlp = spacy.load("en", disable=spacy_disable)
		# Lancaster Stemmer
		stemmer = LancasterStemmer()
		# GB English word list (inc -ise and -ize)
		word_list = toolbox.loadDictionary(basename+"/resources/en_GB-large.txt")
		# Part of speech map file
		tag_map = toolbox.loadTagMap(basename+"/resources/en-ptb_map", args)
	elif args.lang == "de":
		from nltk.stem.snowball import GermanStemmer
		import scripts.cat_rules_de as cat_rules
		import treetaggerwrapper
		treetagger = treetaggerwrapper.TreeTagger(TAGLANG="de",TAGDIR=basename+"/resources/tree-tagger-3.2")
		# Load Tokenizer and other resources
		nlp = spacy.load("de", disable=spacy_disable)
		# German Snowball Stemmer
		stemmer = GermanStemmer()
		# DE word list from hunspell
		word_list = toolbox.loadDictionary(basename+"/resources/de_DE-large.txt")
		# Part of speech map file (not needed for spacy 2.0)
		tag_map = toolbox.loadTagMap(basename+"/resources/de-stts_map", args)
	# Setup output m2 file
	out_m2 = open(args.out, "w")

	print("Processing files...")
	# Open the original and corrected text files.
	with open(args.orig) as orig, open(args.cor) as cor:
		# Process each pre-aligned sentence pair.
		for orig_sent, cor_sent in zip(orig, cor):
			# Markup the parallel sentences with spacy
			proc_orig = toolbox.applySpacy(orig_sent.strip(), nlp, args, treetagger)
			proc_cor = toolbox.applySpacy(cor_sent.strip(), nlp, args, treetagger)
			# Write the original sentence to the output m2 file.
			if args.tok:
				out_m2.write("S "+orig_sent+"\n")
			else:
				proc_orig_tokens = [token.text for token in proc_orig]
				out_m2.write("S "+ " ".join(proc_orig_tokens)+"\n")
			# Identical sentences have no edits, so just write noop.
			if orig_sent.strip() == cor_sent.strip():
				out_m2.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
			# Otherwise, do extra processing.
			else:
				# Output the annotations for orig/cor
				if (args.ann):
				  out_m2.write("O "+toolbox.formatAnnotation(proc_orig)+"\n")
				  out_m2.write("C "+toolbox.formatAnnotation(proc_cor)+"\n")
				# Auto align the parallel sentences and extract the edits.
				auto_edits = align_text.getAutoAlignedEdits(proc_orig, proc_cor, nlp, args)
				# Loop through the edits.
				for auto_edit in auto_edits:
					# Give each edit an automatic error type.
					cat = cat_rules.autoTypeEdit(auto_edit, proc_orig, proc_cor, word_list, tag_map, nlp, stemmer)
					auto_edit[2] = cat
					# Write the edit to the output m2 file.
					out_m2.write(toolbox.formatEdit(auto_edit)+"\n")
			# Write a newline when there are no more edits.
			out_m2.write("\n")
			
if __name__ == "__main__":
	# Define and parse program input
	parser = argparse.ArgumentParser(description="Convert parallel original and corrected text files (1 sentence per line) into M2 format.\nThe default uses Damerau-Levenshtein and merging rules and assumes tokenized text.",
								formatter_class=argparse.RawTextHelpFormatter,
								usage="%(prog)s [-h] [options] -orig ORIG -cor COR -out OUT")
	parser.add_argument("-orig", help="The path to the original text file.", required=True)
	parser.add_argument("-cor", help="The path to the corrected text file.", required=True)
	parser.add_argument("-out",	help="The output filepath.", required=True)						
	parser.add_argument("-lang", choices=["en", "de"], default="en", help="Input language. Currently supported: en (default), de\n")
	parser.add_argument("-lev",	help="Use standard Levenshtein to align sentences.", action="store_true")
	parser.add_argument("-merge", choices=["rules", "all-split", "all-merge", "all-equal"], default="rules",
						help="Choose a merging strategy for automatic alignment.\n"
								"rules: Use a rule-based merging strategy (default)\n"
								"all-split: Merge nothing; e.g. MSSDI -> M, S, S, D, I\n"
								"all-merge: Merge adjacent non-matches; e.g. MSSDI -> M, SSDI\n"
								"all-equal: Merge adjacent same-type non-matches; e.g. MSSDI -> M, SS, D, I")
	parser.add_argument("-tok", help="Tokenize input using spacy tokenizer.", action="store_true")
	parser.add_argument("-ann", help="Output automatic annotation.", action="store_true")
	args = parser.parse_args()
	# Run the program.
	main(args)
