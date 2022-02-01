import tensorflow as tf

from plumber_analysis import pipeline_optimizer, gen_util

def create_optimizer():
    filename = "_optimizer_stats.pb"
    filename = "stats.pb"
    machine_info = {'HOSTNAME': 'Localhost', 'CORES': 32, 'MEMORY': 67430866944,
                    'FILES': [{'PATH': '/mnt/bigdrive/data/train', 'BANDWIDTH':
                               2e9}]}
    machine_info = {'HOSTNAME': 'Localhost', 'CORES': 32, 'MEMORY': 67430866944,
                    'FILES': [{'PATH': '/mnt/bigdrive/data/train',
                                        'BANDWIDTH': 163730000.0}]}
    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(filename)
    optimizer = pipeline_optimizer.DataPipelineOptimizer(plumber,
                                                         calibrate_system=False,
                                                         machine_info=machine_info,
                                                         step_size=None)
    #optimizer = pipeline_optimizer.CostBasedDataPipelineOptimizer(
    #    plumber, calibrate_system=False, step_size=None, min_rate=30)
    return optimizer

def step_par_0():
    print("par 0")
    optimizer = create_optimizer()
    optimizer.apply_parallelism_0()
    return optimizer

def step_cache_0():
    print("cache 0")
    optimizer = create_optimizer()
    print(optimizer.get_cache_summary())
    optimizer.apply_parallelism_0()
    #optimizer.apply_cache_0(False)
    optimizer.apply_cache_0()
    return optimizer

def step_par_1():
    print("par 1")
    optimizer = create_optimizer()
    optimizer.apply_parallelism_0()
    optimizer.apply_cache_0()
    optimizer.update_plumber()
    optimizer.apply_parallelism_1()
    return optimizer

def sweep():
    sweep_range = [step_par_0, step_cache_0, step_par_1]
    sweep_range = [step_cache_0]
    for f in sweep_range:
        optimizer = f()
        dataset = optimizer.instantiate_pipeline()
        gen_util.benchmark_dataset(dataset, time_limit_s=62)

def optimize_default():
    optimizer = create_optimizer()
    #optimizer.roofline("roofline.pdf", ylim="all")
    #print(optimizer.roofline("roofline.pdf"))
    print(optimizer.roofline())
    #print(optimizer.all_N_stats())
    #optimizer.disable_inter_op_parallelism()
    optimizer.apply_optimizations(benchmark_time_s=22, inner_benchmarking=False,
                                  num_optimization_passes=3,
                                  rebench=True)
    dataset = optimizer.instantiate_pipeline()
    #print(optimizer.roofline("roofline_opt.pdf"))
    gen_util.benchmark_dataset(dataset, time_limit_s=62)
    #
    #options = tf.data.Options()
    #gen_util.add_analysis_to_dataset_options(options)
    #dataset = dataset.with_options(options)
    #print("Benchmarking")
    #print("*" * 80)
    #gen_util.drop_caches()
    #gen_util.benchmark_dataset(dataset, time_limit_s=22)

def main():
    #optimize_default()
    sweep()

if __name__ == "__main__":
    main()
