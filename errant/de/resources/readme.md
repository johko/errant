# Resources

## de-stts_map

de-stts_map is a mapping file that converts STTS tags to UD tags. It was copied from the TuebaUDConverter:

https://github.com/bencampbell30/TuebaUdConverter/blob/94aecf24b3b400b1f192e29daa4042cfb19aaa65/TreebankUdConverter/src/PosTransformations.txt

and the mapping PROAV - ADV was added.

Spacy also provides a mapping from STTS -> UD, but (probably among other differences) it maps PTKVZ to PART instead of ADP and VM* to VERB instead of AUX.

## de_DE-large.txt

de_DE-large.txt is a list of German words according to the latest Hunspell dictionary.

Downloaded from: https://www.j3e.de/ispell/igerman98/dict/igerman98-20161207.tar.bz2

Install hunspell and hunspell-tools.

In `igerman98-20161207`:

```
$ make hunspell/de_DE.aff hunspell/de_DE.dic
$ unmunch hunspell/de_DE.dic hunspell/de_DE.aff 2> /dev/null | iconv -f iso-8859-1 -t utf-8 | grep -vP "^[-\t]" | grep -v "/" > de_DE-large.txt
```

