#export TF_CPP_MIN_LOG_LEVEL=0
#export TF_CPP_MAX_VLOG_LEVEL=0

# DO NOT TOUCH
top_level_dir="$(pwd)"
# DO NOT TOUCH

# 96 cores/threads and resnet18 model, big dataset

clear_options() {
	data_dir="gs://mkuchnik_data_eu_west4/data/ILSVRC12/train_TFRecord"
	val_data_dir="gs://mkuchnik_data_eu_west4/data/ILSVRC12/validation_TFRecord"
	#data_dir="/mnt/disks/datadisk/data/"
	#val_data_dir=""
	output_dir="${top_level_dir}/output"
	resnet_depth=18
	batch_size=128
	epochs_per_loop="4"
	mixup_alpha="0.0"
	label_smoothing="0.1"
	echo_factor="0"
	fake_model=False
	input_pipeline_threadpool_size=96
	input_pipeline_default_parallelism=96
	input_pipeline_default_prefetching=100
	optimize_plumber_pipeline=False
	randaugment_num_layers=0
	randaugment_magnitude=5
	# NOTE: we expect each 128 images to have 75MB of memory, so this grows quickly
	# We set to 256 to get roughly 20GB of space.
	echo_shuffle_buffer_size=1024
	force_cache_uncompressed=False
	experiment_prefix=""
	num_epochs=90
	epochs_per_loop_scaler="1.0"
	no_eval=False
	train_script="${top_level_dir}/train.py"
	benchmark_script="${top_level_dir}/benchmark_mlperf.py"
	bench_time_limit_s=60

	fake_model=False
	bench_dir="official_experiments/bench_normal_${bench_time_limit_s}_96_core_96_thread"
	experiment_dir="official_experiments/resnet18_model_96_core_96_thread"
	input_pipeline_threadpool_size=96
	bench_dir="bench/${experiment_dir}"
}

current_options() {
	enable_fast_training
	#enable_small_dataset
	# Only do first training with 1 epoch, since slow
	#num_epochs=10
	num_epochs=5
	num_epochs_scaler="1.0"
}

enable_smoothing() {
	mixup_alpha="0.0"
	label_smoothing="0.1"
}

enable_mixup() {
	mixup_alpha="0.1"
	label_smoothing="0.0"
}

enable_randaugment() {
	randaugment_num_layers=2
}

default_options() {
	echo_factor="0"
	#input_pipeline_threadpool_size=48
	input_pipeline_threadpool_size=96
	input_pipeline_default_parallelism=96
	input_pipeline_default_prefetching=100
	optimize_plumber_pipeline=False
	#enable_mixup
	#enable_randaugment
}

increase_threadpool_size() {
	input_pipeline_threadpool_size=96
}

autotune_options() {
	echo_factor="0"
	input_pipeline_default_parallelism=-1
	input_pipeline_default_prefetching=-1
	#input_pipeline_default_prefetching=0
	optimize_plumber_pipeline=False
}

static_options() {
	input_pipeline_default_parallelism=96
	input_pipeline_default_prefetching=100
	#enable_small_dataset
	#force_cache_uncompressed
}

naive_options() {
	echo_factor="0"
	input_pipeline_threadpool_size=1
	input_pipeline_default_parallelism=1
	input_pipeline_default_prefetching=0
	optimize_plumber_pipeline=False
}

enable_optimizations() {
	optimize_plumber_pipeline=True
}

enable_small_dataset() {
	data_dir="gs://mkuchnik_data_eu_west4/data/ILSVRC12/validation_TFRecord/validation-*"
}

enable_big_dataset() {
	data_dir="gs://mkuchnik_data_eu_west4/data/ILSVRC12/train_TFRecord"
}

enable_cache_uncompressed() {
	force_cache_uncompressed=True
}

enable_fast_training() {
	epochs_per_loop=1
	num_epochs=5
}

test_plumber() {
	default_options
	#enable_smoothing
	#enable_small_dataset
	enable_optimizations
}

test_static() {
	static_options
}

test_optimal_static() {
	static_options
	increase_threadpool_size
}

test_autotune() {
	default_options
	#enable_smoothing
	#enable_small_dataset
	autotune_options
}

test_optimal_autotune() {
	static_options
	increase_threadpool_size
}

run_experiment_naive() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	naive_options
	#num_epochs=1
	#epochs_per_loop_scaler="0.1"
	experiment_prefix="${experiment_dir}/naive/"
	for e in 0
	do
		echo_factor=$e
		run_training_bench_heuristic_training_echo_factor
	done
}

run_experiment_naive_bench() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	naive_options
	experiment_prefix="${bench_dir}/naive/"
	for e in 0
	do
		echo_factor=$e
		run_benchmark_pipeline
	done
}

run_experiment_static() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_static
	experiment_prefix="${experiment_dir}/static/"
	for e in 0
	do
		echo_factor=$e
		run_training_bench_heuristic_training_echo_factor
	done
}

run_experiment_static_bench() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_static
	experiment_prefix="${bench_dir}/static/"
	for e in 0
	do
		echo_factor=$e
		run_benchmark_pipeline
	done
}

run_experiment_optimal_static() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_optimal_static
	experiment_prefix="${experiment_dir}/opt_static/"
	for e in 0
	do
		echo_factor=$e
		run_training_bench_heuristic_training_echo_factor
	done
}

run_experiment_optimal_static_bench() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_optimal_static
	experiment_prefix="${bench_dir}/opt_static/"
	for e in 0
	do
		echo_factor=$e
		run_benchmark_pipeline
	done
}

run_experiment_autotune() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_autotune
	experiment_prefix="${experiment_dir}/autotune/"
	for e in 0
	do
		echo_factor=$e
		run_training_bench_heuristic_training_echo_factor
	done
}

run_experiment_autotune_bench() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_autotune
	experiment_prefix="${bench_dir}/autotune/"
	for e in 0
	do
		echo_factor=$e
		run_benchmark_pipeline
	done
}

run_experiment_optimal_autotune() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_autotune
	test_optimal_autotune
	experiment_prefix="${experiment_dir}/opt_autotune/"
	for e in 0
	do
		echo_factor=$e
		run_training_bench_heuristic_training_echo_factor
	done
}

run_experiment_optimal_autotune_bench() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_autotune
	test_optimal_autotune
	experiment_prefix="${bench_dir}/opt_autotune/"
	for e in 0
	do
		echo_factor=$e
		run_benchmark_pipeline
	done
}

run_experiment_plumber() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_plumber
	experiment_prefix="${experiment_dir}/plumber/"
	for e in 0
	do
		echo_factor=$e
		run_training_bench_heuristic_training_echo_factor
	done
}

run_experiment_plumber_bench() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_plumber
	experiment_prefix="${bench_dir}/plumber/"
	for e in 0
	do
		echo_factor=$e
		run_benchmark_pipeline
	done
}

run_experiment_plumber_find_best() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_plumber
	experiment_prefix="${experiment_dir}/plumber_find_best/"
	for e in 0
	do
		echo_factor=$e
		PLUMBER_FIND_BEST_PIPELINE=True run_training_bench_heuristic_training_echo_factor
	done
}

run_experiment_plumber_find_best_bench() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_plumber
	experiment_prefix="${bench_dir}/plumber_find_best/"
	for e in 0
	do
		echo_factor=$e
		PLUMBER_FIND_BEST_PIPELINE=True run_benchmark_pipeline
	done
}

run_experiment_plumber_cache_path() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_plumber
	experiment_prefix="${experiment_dir}/plumber_cache_path/"
	for e in 0
	do
		echo_factor=$e
		# TODO(mkuchnik): Prevents cache from being sticky?
		force_cache_uncompressed=True run_training_bench_heuristic_training_echo_factor
	done
}

run_experiment_plumber_cache_path_bench() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_plumber
	experiment_prefix="${bench_dir}/plumber_cache_path/"
	for e in 0
	do
		echo_factor=$e
		# TODO(mkuchnik): Prevents cache from being sticky?
		force_cache_uncompressed=True run_benchmark_pipeline
	done
}


run_experiment_plumber_no_opt() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_plumber
	experiment_prefix="${experiment_dir}/plumber_no_opt/"
	for e in 0
	do
		echo_factor=$e
		PLUMBER_OVERRIDE_PRESETS=False run_training_bench_heuristic_training_echo_factor
	done
}

run_experiment_plumber_no_caching() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_plumber
	experiment_prefix="${experiment_dir}/plumber_no_caching/"
	for e in 0
	do
		echo_factor=$e
		PLUMBER_APPLY_CACHING=False run_training_bench_heuristic_training_echo_factor
	done
}

run_experiment_plumber_no_caching_1_step() {
	list_python_programs
	kill_python_programs
	kill_python_programs
	list_python_programs
	clear_options
	current_options
	default_options
	test_plumber
	experiment_prefix="${experiment_dir}/plumber_no_caching_1_step/"
	for e in 0
	do
		echo_factor=$e
		PLUMBER_APPLY_CACHING=False PLUMBER_APPLY_NUM_STEPS=1 run_training_bench_heuristic_training_echo_factor
	done
}

clear_options

. run_def.sh

list_python_programs
kill_python_programs

#default_options
#test_plumber
#run_benchmark_pipeline

## End-to-Ends
#run_experiment_optimal_autotune
#run_experiment_optimal_static
run_experiment_autotune
run_experiment_static
run_experiment_plumber_find_best
run_experiment_naive

## Benches
run_experiment_naive_bench
run_experiment_static_bench
run_experiment_autotune_bench
run_experiment_plumber_find_best_bench
