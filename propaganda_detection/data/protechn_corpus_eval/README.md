Contents

1. About the Propaganda Techniques Corpus (PTC)
2. Tasks
3. Data format
4. Evaluation script
5. Citation 

About the Propaganda Techniques Corpus (PTC)
--------------------------------------------

PTC is a corpus of propagandistic techniques at fine-grained level. 

The corpus includes 451 articles (350k tokens) from 48 news outlets. It was 
manually-annotated by six professional annotators (both unitized and labeled) 
considering 18 propaganda techniques:

* Loaded Language
* Name Calling,Labeling
* Repetition
* Exaggeration,Minimization
* Doubt
* Appeal to fear-prejudice
* Flag-Waving
* Causal Oversimplification
* Slogans
* Appeal to Authority
* Black-and-White Fallacy
* Thought-terminating Cliches
* Whataboutism
* Reductio ad Hitlerum
* Red Herring
* Bandwagon
* Obfuscation,Intentional Vagueness,Confusion
* Straw Men


Tasks
--------------------------------------------
PTC enables for the development of automatic models for propaganda tecniques 
identification (multi-class setting) and the binary identification of 
propagandistic text fragments (binary setting). See the paper in section 
"Citation" for further details 


Data format
--------------------------------------------
The corpus includes one plain-text and one tab-separated file per article. 
The former contains the row contents of the article, as extracted with the 
newspaper3k library. The latter contains one propaganda technique per line with 
the following information: 

id   technique    begin_offset     end_offset

where id is the identifier of the article, technique is one out of the 18
techniques, begin_offset is the character where the covered span begins and 
end_offset is the character where the covered span ends.

The naming of the pair of files is:
- article[unique_id].txt for the plain-text file 
- article[unique_id].labels.tsv for the annotations files 

We include three subfolders: train (293 articles), dev (57 articles), and test
(101 articles).

Evaluation script
--------------------------------------------

We also provide an evaluation script "proptech_scorer.py". It computes micro-
averaged precision, recall, and F-measure for all classess as well as for 
each class in isolation.

The scorer requires file propaganda-techniques-names.txt (included) and can be 
run as follows:

python3 proptech_scorer.py -s [prediction_file] -r [gold_folder]

As an example, we provide a "prediction_file" which concatenates all the 
tsv files in the training partitions and run it as follows:

===
$ python3 proptech_scorer.py -s submission.tsv -r train/
2019-03-05 13:13:26,034 - INFO - Checking user prediction file submission.tsv against gold folder train/
2019-03-05 13:13:26,158 - INFO - Scoring user prediction file submission.tsv against gold file train/
2019-03-05 13:13:26,597 - INFO - Precision=5114.000000/5114=1.000000	Recall=5114.000000/5114=1.000000
2019-03-05 13:13:26,597 - INFO - F1=1.000000
2019-03-05 13:13:26,598 - INFO - F1-Appeal_to_Authority=1.000000
2019-03-05 13:13:26,599 - INFO - F1-Appeal_to_fear-prejudice=1.000000
2019-03-05 13:13:26,600 - INFO - F1-Bandwagon=1.000000
2019-03-05 13:13:26,600 - INFO - F1-Black-and-White_Fallacy=1.000000
2019-03-05 13:13:26,601 - INFO - F1-Causal_Oversimplification=1.000000
2019-03-05 13:13:26,602 - INFO - F1-Doubt=1.000000
2019-03-05 13:13:26,603 - INFO - F1-Exaggeration,Minimisation=1.000000
2019-03-05 13:13:26,603 - INFO - F1-Flag-Waving=1.000000
2019-03-05 13:13:26,604 - INFO - F1-Loaded_Language=1.000000
2019-03-05 13:13:26,605 - INFO - F1-Name_Calling,Labeling=1.000000
2019-03-05 13:13:26,606 - INFO - F1-Obfuscation,Intentional_Vagueness,Confusion=1.000000
2019-03-05 13:13:26,606 - INFO - F1-Red_Herring=1.000000
2019-03-05 13:13:26,607 - INFO - F1-Reductio_ad_hitlerum=1.000000
2019-03-05 13:13:26,608 - INFO - F1-Repetition=1.000000
2019-03-05 13:13:26,608 - INFO - F1-Slogans=1.000000
2019-03-05 13:13:26,609 - INFO - F1-Straw_Men=1.000000
2019-03-05 13:13:26,610 - INFO - F1-Thought-terminating_Cliches=1.000000
2019-03-05 13:13:26,610 - INFO - F1-Whataboutism=1.000000
===

Citation 
--------------------------------------------
please cite the paper "ANONYMOUS" when using this corpus.
