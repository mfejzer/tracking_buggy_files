# tracking_buggy_files
This repository contains scripts to process two datasets, feature preparation code and implementation of algorithms from publication "Tracking Buggy Files: New Efficient Adaptive Bug Localization method".
The java-ast-extractor contains 4 programs enriching source code files with ast trees.
The ast trees are stored as git notes per each source file.
The java 8 and apache maven are required to compile java-ast-extractor.
Rest of scripts require python 3 and python 2.

# How to replicate dataset - example for AspectJ project, using already existing git notes
* Download original dataset
* Clone repository https://bitbucket.org/mfejzer/tracking_buggy_files_aspectj_dataset/
* Fetch git notes containing ast trees and import graphs
```
git fetch origin refs/notes/commits:refs/notes/commits
git fetch origin refs/notes/tokenized_counters:refs/notes/tokenized_counters
git fetch origin refs/notes/graph:refs/notes/graph
```
* Convert project files from xml to json:
```
./process_bug_reports.py AspectJ.xml ../tracking_buggy_files_aspectj_dataset/ aspectj_base.json
./fix_and_augment.py aspectj_base.json ../tracking_buggy_files_aspectj_dataset/ > aspectj_aug.json
./pick_bug_freq.py aspectj_aug.json ../tracking_buggy_files_aspectj_dataset/ > aspectj.json
```
* Calculate features - result files will be stored using prefix "aspectj":
```
./create_ast_cache.py ../tracking_buggy_files_aspectj_dataset/ aspectj.json aspectj
./vectorize_ast.py aspectj.json aspectj
./vectorize_enriched_api.py aspectj.json aspectj
./convert_tf_idf.py aspectj.json aspectj
./calculate_feature_3.py aspectj.json aspectj
./retrieve_features_5_6.py aspectj.json aspectj
./calculate_notes_graph_features.py aspectj.json aspectj ../tracking_buggy_files_aspectj_dataset/
./calculate_vectorized_features.py aspectj.json aspectj
./save_normalized_fold_dataframes.py aspectj.json aspectj
```
# How to replicate adaptive method results
Example for AspectJ project, using same data prefix as feature calculation
```
./load_data_to_joblib_memmap.py aspectj
./train_adaptive.py aspectj
```
# Dataset repositories containing git notes:
* https://bitbucket.org/mfejzer/tracking_buggy_files_aspectj_dataset/
* https://bitbucket.org/mfejzer/tracking_buggy_files_birt_dataset/
* https://bitbucket.org/mfejzer/tracking_buggy_files_eclipse_platform_ui_dataset/
* https://bitbucket.org/mfejzer/tracking_buggy_files_jdt_dataset/
* https://bitbucket.org/mfejzer/tracking_buggy_files_swt_dataset/
* https://bitbucket.org/mfejzer/tracking_buggy_files_tomcat_dataset/

# How to prepare eclipse project from single version buglocator dataset
Example using data prefix eclipse_311 and eclipse 311 sources in "sources" dir.
Requires compiled java-ast-extractor.
Eclipse 311 sources (eclipse-sourceBuild-srcIncluded-3.1.zip) can be downloaded from http://archive.eclipse.org/eclipse/downloads/drops/R-3.1-200506271435/
```
./process_buglocator.py EclipseBugRepository.xml EclipseBugRepository.json
java -jar java-ast-extractor-source-snapshot.jar sources extract.txt
./tokenize_buglocator_source.py extract.txt tokenized_extract.txt
./vectorize_buglocator_source.py EclipseBugRepository.json tokenized_extract.txt eclipse_311
./vectorize_buglocator_enriched_api.py extract.txt eclipse_311
./calculate_buglocator_time_features.py EclipseBugRepository.json eclipse_311
./calculate_buglocator_feature_3.py EclipseBugRepository.json eclipse_311
./calculate_buglocator_graph_features.py extract.txt eclipse_311
./calculate_buglocator_features.py EclipseBugRepository.json eclipse_311
./save_normalized_fold_dataframes_buglocator.py EclipseBugRepository.json eclipse_311
```
