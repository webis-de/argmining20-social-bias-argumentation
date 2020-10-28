# Function to generate a glove model
# Extects the following parameter (in that order): corpus_path, output_path, output_prefix
generate_glove_model () {
    echo ""
    echo -e "\e[1;31m================================================================================\e[0m"
    echo -e "\e[1;31mWorking on $3...\e[0m"

    make

    PREFIX=$3
    OUTPUT_DIR=$2
    CORPUS=$1

    VOCAB_FILE=$OUTPUT_DIR/$PREFIX-vocab.txt
    COOCCURRENCE_FILE=$OUTPUT_DIR/$PREFIX-cooccurrence.bin
    COOCCURRENCE_SHUF_FILE=$OUTPUT_DIR/$PREFIX-cooccurrence.shuf.bin
    BUILDDIR=build
    SAVE_FILE=$OUTPUT_DIR/$PREFIX-vectors
    VERBOSE=2
    MEMORY=4.0
    VOCAB_MIN_COUNT=5
    VECTOR_SIZE=300
    MAX_ITER=100
    WINDOW_SIZE=15
    BINARY=2
    NUM_THREADS=8
    X_MAX=10
    if hash python 2>/dev/null; then
        PYTHON=python
    else
        PYTHON=python3
    fi

    echo
    echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
    $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
    echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
    $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
    echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
    $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE


    echo "$ $PYTHON eval/python/evaluate.py"
    $PYTHON eval/python/evaluate.py --vocab_file $VOCAB_FILE --vectors_file $SAVE_FILE.txt
}
