# Assessing Social Bias in Argumentation

This repository contains code to reproduce the results of the paper "Argument from Old Man’s View: Assessing Social Bias in Argumentation", as presented at the ArgMining'20 workshop. Please find the full reference below:

```
@inproceedings{spliethoever-2020,
    title     = "Argument from Old Man's View: Assessing Social Bias in Argumentation",
    author    = {Splieth{\"o}ver, Maximilian and Wachsmuth, Henning},
    booktitle = Proceedings of the 7th Workshop on Argument Mining,
    month     = aug,
    year      = 2020
}
```

## Reproduce the results
The code in the repository was tested and the results created with _Python version: 3.6.10 (Linux)_.

### Install the requirements
In order to execute any of the scripts, you'll need to install certain python packages. Those packages are specified in the `requirements.txt` file, alongside their versions. Install them using `pip`.
```shell
$ pip install -r requirements.txt
```

Further, some of the scripts build on the `spaCy` language model. Thus, it needs to be installed as well.
```shell
$ python -m spacy download en_core_web_sm
```


### Download the data and baseline models
The study builds on different corpora and pre-trained word embedding models. Please follow the table below to download the data (from the _URL_ column) and place the specified files (in the _Files_ column) into the specified directories (in the _Directory_ column). All original references for the corpora can be found in the full paper.

| **Corpus/Model** | **URL** | **Files** | **Extract** | **Directory** |
|-|-|-|-|-|
| Internet Argument Corpus (IAC) v2 | https://nlds.soe.ucsc.edu/iac2 | `convinceme_2016_05_18.sql`<br>`createdebate_released_2016_05_18.sql`<br>`fourforums_2016_05_18.sql` | Yes | `data/iac/` |
| Webis-CMV-20 | https://zenodo.org/record/3778298 | `threads.jsonl` | Yes | `data/` |
| debate.org corpus | http://www.cs.cornell.edu/~esindurmus/ddo.html | `debates.json`<br>`users.json` | No | `data/` |
| GloVe CommonCrawl (840B) | https://nlp.stanford.edu/projects/glove/ | `glove.840B.300d.txt` | Yes | `word_vectors/` |
| Numberbatch 19.08 (English only) | https://github.com/commonsense/conceptnet-numberbatch | `numberbatch-en.txt` | Yes | `word_vectors/` |


### Data preprocessing
For each corpus, different preprocessing steps are required, e.g. text extraction and cleaning. The table below lists the scripts to be run.

| **Corpus** | **Scripts** | **Description** |
|-|-|-|
| Internet Argument Corpus (IAC) v2 | `import_iac_to_mysql.sh`<br>`run_prepare_iac_data.sh` | For this corpus, two scripts are necessary. The first imports the dumps into a MySQL database. Note that [MySQL](https://www.mysql.com/) needs to be installed for this to work. Also, the python packages [`sqlalchemy`](https://www.sqlalchemy.org/) and [`mysqlclient`](https://pypi.org/project/mysqlclient/) are required. Due to the size of the dumps, this process can take a while (the fourforums dump can take hours even). The second script then exports the dumps into text files and runs all preprocessing steps on it. In both files, it might be necessary to change some parameters, e.g. the MySQL username, password and database address. Refer to the [`prepare_iac_data.py`](prepare_iac_data.py) file for preprocessing details. |
| Webis-CMV-20 | `run_prepare_cmv_data.sh` | Bash scripts that executes the python preprocessing script with predefined parameters. Refer to the [`prepare_cmv_data.py`](prepare_cmv_data.py) file for preprocessing details. |
| debate.org corpus | `run_prepare_ddo_data.sh` | Bash scripts that executes the python preprocessing script with predefined parameters. Due to the size of the corpus, the script tries to do as much work as possible on multiple cores; those parameters might need to be changed. Refer to the [`prepare_cmv_data.py`](prepare_cmv_data.py) file for preprocessing details. |


### Create custom embedding models
The custom embedding models are created using the GloVe algorithm. Thus, the data needs to be changed into a format the GloVe library understands. Running the data preprocessing scripts listed above is also required beforehand, as the GloVe preprocessing builds on the output of them. In addition to preparing the text files, the preprocessing scripts also split the text into a number of subsets that will later be used to calculate the stability of the results of each embedding evaluation.

After preparing the texts, each corpus has a separate script that runs the GloVe algorithm. You can find a reference to both scripts for each corpus in the table below. Scripts to generate the GloVe embeddings are expected to be executed from within the `GloVe/` directory. Currently, the script will use 8 CPU threads to generate the models. You can change the `NUM_THREADS` parameter in the [`GloVe/generate_glove_model.sh`](GloVe/generate_glove_model.sh) to increase/decrease this value.

**Note**: The debate.org dataset additionally requires a definition of which subgroups to extract from the prepared data. The parameters used in the original paper are specified in the [`data/groups_of_interest.json`](data/groups_of_interest.json) file and are used by default.

**Note**: For some of the smaller corpora, such as createdebate, it is not possible to train embedding models for the splits. The scripts will print an error message at the evaluation of those models and continue with the next one.

| Name | GloVe preprocessing script | GloVe generation script |
|-|-|-|
| Internet Argument Corpus (IAC) v2 | `run_prepare_iac_glove_input.sh` | `GloVe/generate_iac_models.sh` |
| Webis-CMV-20 | `run_prepare_cmv_glove_input.sh` | `GloVe/generate_cmv_models.sh` |
| debate.org corpus | `run_prepare_ddo_glove_input.sh` | `GloVe/generate_ddo_models.sh` |

Further, after the embedding models are generated, you'll need to add length information to the vector files in order to adhere to the word2vec format. To do this, use the following snippet on each of the model files. You can identify them in the `output/GloVe/` directory by their file name ending `-vectors.txt`.
```shell
$ # Replace `/path/to/file` with the path to the vector file
$ FILEPATH=/path/to/file; sed -i "1s/^/$(echo $(cat ${FILEPATH} | wc -l)) 300\n/" $FILEPATH
```

You can also use the script `convert_glove_to_word2vec_format.sh`. This will run the command above for all vector files in the `output/glove/` and `output/glove/splits/` directory. But be careful to only run this once, as it won't check if the vector file is already in the proper format and thus may corrupt the file if run multiple times on the same file.

**Note**: To evaluate the original GloVe embedding model you also need to apply the conversion snippet above to it.


### Social bias evaluation of (custom) embedding models
The pre-trained embedding models are evaluated using the WEAT formula and word list. You can find the re-implementation at [`sbeval/weat_test.py`](sbeval/weat_test.py) and the tests comparing it to the original results at [`sbeval/tests/`](sbeval/tests/). To evaluate all baseline and custom generated embedding models (including the ones of the smaller subsets), execute the `run_all_embedding_bias_evaluations.sh` script. For single model evaluations, refer to the `run_embedding_bias_evaluation.sh` script. The evaluation results can then be found in the `output/embedding_model_evaluation/` directory.

The stability/reliability of the results for each embedding model is represented by the standard deviation of the subsets' WEAT results, as explained in the paper, and calculated manually.

**Note**: Due to the random initialization of the GloVe models, it is possible that this evaluation outputs different results to the ones reported in the paper.


### Co-occurrence analyses
The paper describes two different co-occurrence analyses. First, all words around the group identity terms are analysed (lexical corpus evaluation). Second, the co-occurences of the group identity and associations words in the same sentence are being counted (weat co-occurrence analysis). To reproduce the results for both, execute the scripts in the table below. Note that the scripts try to do as much work as possible on multiple cores; those parameters might need to be changed in both scripts.

| **Name** | **Script** | **Output directory** | **Description** |
|-|-|-|-|
| Lexical corpus evaluation | `run_lexical_corpus_evaluation.sh` | `output/lexical_corpus_evaluation/` | This analysis uses the group identity words of the WEAT to extract the most common co-occurring terms in a given window size. As a side product, it will also output the total number of occurrences of those identity terms and the number of posts that include at least one of them. The file [`data/lexical-analysis-lexicon.json`](`data/lexical-analysis-lexicon.json`) defines the lists that are used. Group identity terms of the WEAT-5 are excluded from this list as they are equal to the WEAT-4 lists. The script further uses the positive and negative word lists from [here](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) to evaluate the number of co-occurrences in a given window size are either positive or negative. |
| Weat Co-occurrence analysis | `run_weat_cooccurrence_analysis.sh` | `output/weat_cooccurrence_analysis/` | In contrast to the analysis above, this script will look at the specific co-occurrences of the WEAT group identity words and the attribute terms in the same sentence. |
| Weat Co-occurrence analysis counts | `run_accumulate_cooccurrence_counts.sh` | - | Takes the output file of the analysis script above as input file and accumulates the counts for the different WEAT lexicons. It will generate an output to the console and have no output file. Those are also the counts you can find in the paper. You will need to adapt the script to point to the correct input file. |


## Notes on the WEAT re-implementation
Since, at the time of conducting the experiments, there was no official WEAT implementation available publicly, we re-implemented the approach from the information available in the original paper and its supplementary material (you can find both [here](https://science.sciencemag.org/content/356/6334/183)). While the evaluation results of the pre-trained word embeddings models with our implementation is not exactly the same, we attribute those smaller changes to implementation details. You can run the score replications with `$ python -m unittest ddo.tests.weat_score_replication_w2v` for word2vec embedding model and `$ python -m unittest ddo.tests.weat_score_replication_glove` for GloVe embedding model. Passing tests are within a boundary specified in the [`sbeval/constants.py`](sbeval/constants.py) file.
