acronym_expanders=(
	"Doc2VecPerAcronym"
	"ContextVector"
	"LDAEXP"
	"DualEncoder"
	"EXPALBERT"
	"SBE"
	)

acronym_expanders_aux=(
	"Cossim"
	"SVM"
	"RF"
	"LR"
	"Keras"
	)

text_representators=(
	"TFIDF"
	"LDA"
	"Doc2Vec"
	"Doc2VecExp"
	#"ALBERT"
	"ContextVector"
	"Concat1"
	"Concat2"
	#"NGramsContextVector"
	"NewLocality"
	"LocalityTFIDF"
	"LocalityContext"
	)

for i in "${acronym_expanders[@]}"; do
	echo "Acronym expander : $i"
	python french_benchmark.py $i
done

for i in "${acronym_expanders_aux[@]}"; do
echo "Acronym Epender aux : $i"
	for j in "${text_representators[@]}"; do
		echo "Text Representator : $j"
		python french_benchmark.py $i $j
	done
done
