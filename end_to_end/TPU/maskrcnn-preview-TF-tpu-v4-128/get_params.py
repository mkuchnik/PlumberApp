"""Get Plumber recommendation from a traced pipeline.

Takes as input the 'stats.pb' in the current directory.
"""

import plumber_analysis.pipeline_optimizer_wrapper
import plumber_analysis.config

is_fast = False
optimizer = plumber_analysis.pipeline_optimizer_wrapper.step_par_2(is_fast=is_fast)

experiment_params = optimizer.experiment_params()
performance_params = optimizer.get_performance_parameters()
print("Experimental params:\n{}".format(experiment_params))
print("Plumber found parameters:\n{}".format(performance_params))

dataset = optimizer.instantiate_pipeline()
dataset = plumber_analysis.pipeline_optimizer_wrapper.apply_default_options(dataset, override_presets=True)
ret = plumber_analysis.pipeline_optimizer_wrapper._benchmark_dataset(dataset, time_limit_s=62)
print(ret)
