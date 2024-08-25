~~~bash

1_done. cd ./examples/scripts
2_done. bash generated_commands_humaneval.sh # run experiments
3_done. bash judge_all_config.sh # judge the agent score
4_done. bash token_count_all_config.sh # count the token map for graph analysis
5_x. bash generated_calc_perturbation_results.sh # no need currently
6_done. bash generated_calc_graph_attributes.sh
7_done. bash {arch}/{task}/generated_aggregate_statistics.sh
8. python calc_regression.py


# for the safetest, we still need to judge the score from perturbation config
3. bash judge_all_config.sh
3.1. bash judge_safe_score_for_perturbation

# use print_safe_score_post_edit.py for better read results
option. 
```
cd output
python print_safe_score_post_edit.py
```

~~~