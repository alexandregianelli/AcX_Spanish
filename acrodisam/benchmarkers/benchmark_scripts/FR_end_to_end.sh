 #!/bin/sh
text_representators=(
	"Doc2Vec"
	"ContextVector"
	"SBE"
	"Concat1"
	"Concat2"
	"LDA"
	"TFIDF"
	"NewLocality"
	)

acronym_expander=(
	"LR"
	"SVM"
	"Cossim"
	"RF"
	)

for i in "${text_representators[@]}"; do
	for j in "${acronym_expander[@]}"; do
		echo "python french_benchmark.py FR FR $i $j"
		python benchmark_french.py FR FR $i $j
	done
done
