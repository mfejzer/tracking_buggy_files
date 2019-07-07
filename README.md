# tracking_buggy_files
This repository contains dataset, feature preparation code and implementation of algorithms from publication "Tracking Buggy Files: Two New Efficient Adaptive Bug Localization methods"

# How to replicate dataset - example for AspectJ project
* Download original dataset
* Clone repositoriy
* Convert project files from xml to json:
```
$ ./process_bug_reports.py AspectJ.xml ../tracking_buggy_files_aspectj_dataset/ aspectj_base.json
$ ./fix_and_augment.py aspectj_base.json ../tracking_buggy_files_aspectj_dataset/ > aspectj.json
```
* Calculate features:
```
./create_ast_cache.py ../tracking_buggy_files_aspectj_dataset/ aspectj.json aspectj
./vectorize_ast.py aspectj.json aspectj
./vectorize_enriched_api.py aspectj.json aspectj
./convert_tf_idf.py aspectj.json aspectj
./calculate_feature_3.py aspectj.json aspectj
./retrieve_features_5_6.py aspectj.json aspectj
./calculate_notes_graph_features.py aspectj.json aspectj ../tracking_buggy_files_aspectj_dataset/
./calculate_vectorized_features.py aspectj.json aspectj
```

# Dataset repositories containing git notes:
* https://bitbucket.org/mfejzer/tracking_buggy_files_aspectj_dataset/
* https://bitbucket.org/mfejzer/tracking_buggy_files_birt_dataset/
* https://bitbucket.org/mfejzer/tracking_buggy_files_eclipse_platform_ui_dataset/
* https://bitbucket.org/mfejzer/tracking_buggy_files_jdt_dataset/
* https://bitbucket.org/mfejzer/tracking_buggy_files_swt_dataset/
* https://bitbucket.org/mfejzer/tracking_buggy_files_tomcat_dataset/

