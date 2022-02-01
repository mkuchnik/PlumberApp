import tensorflow as tf

from plumber_analysis import pipeline_optimizer, gen_util

def create_optimizer():
    filename = "_optimizer_stats.pb"
    filename = "stats.pb"
    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(filename)
    optimizer = pipeline_optimizer.DataPipelineOptimizer(plumber,
                                                         calibrate_system=False,
                                                         step_size=None)
    optimizer = pipeline_optimizer.CostBasedDataPipelineOptimizer(
        plumber, calibrate_system=False, step_size=None, min_rate=30)
    return optimizer

optimizer = create_optimizer()
print(optimizer.get_performance_parameters())
#optimizer.roofline("roofline.pdf", ylim="all")
print(optimizer.roofline("roofline.pdf"))
print(optimizer.all_N_stats())
#optimizer.disable_inter_op_parallelism()
optimizer.apply_optimizations()
print(optimizer.roofline("roofline_opt.pdf"))
dataset = optimizer.instantiate_pipeline()
#
#options = tf.data.Options()
#gen_util.add_analysis_to_dataset_options(options)
#dataset = dataset.with_options(options)
#print("Benchmarking")
#print("*" * 80)
#gen_util.benchmark_dataset(dataset, time_limit_s=62)
