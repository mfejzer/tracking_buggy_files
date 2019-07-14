# tracking_buggy_files
This repository contains dataset, feature preparation code and implementation of algorithms from publication "Tracking Buggy Files: Two New Efficient Adaptive Bug Localization methods"

# How to replicate dataset - example for AspectJ project
* Download original dataset
* Clone repositoriy https://bitbucket.org/mfejzer/tracking_buggy_files_aspectj_dataset/
* Fetch git notes containing ast trees and import graphs
```
git fetch origin refs/notes/commits:refs/notes/commits
git fetch origin refs/notes/graph:refs/notes/graph
```
* Convert project files from xml to json:
```
$ ./process_bug_reports.py AspectJ.xml ../tracking_buggy_files_aspectj_dataset/ aspectj_base.json
$ ./fix_and_augment.py aspectj_base.json ../tracking_buggy_files_aspectj_dataset/ > aspectj.json
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
# How to replicate results - example for AspectJ project, using same data prefix as feature calculation
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

