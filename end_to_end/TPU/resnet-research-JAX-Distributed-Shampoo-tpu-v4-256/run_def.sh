list_python_programs() {
	ps aux | grep python | grep -v "grep python" | awk '{print $2}'
}

kill_python_programs() {
	ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
	sleep 1
	ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
	sleep 1
	ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
	sleep 1
	ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
	sleep 1
	ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
	sleep 1
}

run_training_fake() {
	python3 train.py --data_dir=${data_dir} --val_data_dir=${val_data_dir} --output_dir=$output_dir \
		--epochs_per_loop=${epochs_per_loop} --fake_data=True --no_eval=True
}

run_training() {
	python3 train.py --data_dir=${data_dir} --val_data_dir=${val_data_dir} --output_dir=$output_dir \
		--epochs_per_loop=${epochs_per_loop} --xprof=True --cache=False \
		--resnet_depth=${resnet_depth} --no_eval=True --input_pipeline_default_parallelism=64
}

run_training_slow() {
	python3 train.py --data_dir=${data_dir} --val_data_dir=${val_data_dir} --output_dir=$output_dir \
		--epochs_per_loop=${epochs_per_loop} --xprof=False --cache=False --resnet_depth=${resnet_depth} --no_eval=True --input_pipeline_default_parallelism=1
}

run_training_bench() {
	python3 train.py --data_dir=${data_dir} --val_data_dir=${val_data_dir} --output_dir=$output_dir \
		--epochs_per_loop=${epochs_per_loop} --xprof=False --cache=False --resnet_depth=${resnet_depth} --no_eval=True --input_pipeline_default_parallelism=64
}

# AUTOTUNE

run_training_bench_autotune() {
	python3 train.py --data_dir=${data_dir} --val_data_dir=${val_data_dir} --output_dir=$output_dir \
		--epochs_per_loop=${epochs_per_loop} --xprof=False --cache=False --resnet_depth=${resnet_depth} --no_eval=True --input_pipeline_default_parallelism=-1
}

run_training_bench_autotune_prefetch() {
	python3 train.py --data_dir=${data_dir} --val_data_dir=${val_data_dir} --output_dir=$output_dir \
		--epochs_per_loop=${epochs_per_loop} --xprof=False --cache=False --resnet_depth=${resnet_depth} --no_eval=True --input_pipeline_default_parallelism=-1 \
		--input_pipeline_default_prefetching=-1
}

# HEURISTIC

run_training_bench_heuristic() {
	python3 train.py --data_dir=${data_dir} --val_data_dir=${val_data_dir} --output_dir=$output_dir \
		--epochs_per_loop=${epochs_per_loop} --xprof=False --cache=False --resnet_depth=${resnet_depth} --no_eval=True --input_pipeline_default_parallelism=64 \
		--optimize_plumber_pipeline=False \
		--input_pipeline_default_prefetching=100
}

run_training_bench_heuristic_training() {
	experiment_name=${experiment_prefix}HEURISTIC_TRAINING
	mkdir -p ${experiment_name}
	pushd ${experiment_name}
	#JAX_DEBUG_NANS=True \
	python3 ../train.py --data_dir=${data_dir} --val_data_dir=${val_data_dir} --output_dir=$output_dir \
		--epochs_per_loop=${epochs_per_loop} --xprof=False --cache=False --resnet_depth=${resnet_depth} --no_eval=False
		--optimize_plumber_pipeline=False \
		--input_pipeline_default_prefetching=${input_pipeline_default_prefetching} \
		--input_pipeline_default_parallelism=${input_pipeline_default_parallelism} \
		--input_pipeline_threadpool_size=${input_pipeline_threadpool_size} \
		--optimizer=momentum \
		--learning_rate=0.1 \
		--momentum=0.9 \
		--num_epochs=90 \
		--weight_decay=1e-4 \
		--mixup_alpha=${mixup_alpha} \
		--label_smoothing=${label_smoothing} \
		--cache=False \
		--optimize_plumber_pipeline=${optimize_plumber_pipeline} \
		--randaugment_num_layers=${randaugment_num_layers} \
		--randaugment_magnitude=${randaugment_magnitude} \
		--fake_model=${fake_model} \
		--force_cache_uncompressed=${force_cache_uncompressed} \
		2>&1 | tee log.txt
	popd
}

run_training_bench_heuristic_training_echo_factor() {
	experiment_name=${experiment_prefix}HEURISTIC_TRAINING_echo_${echo_factor}
	mkdir -p ${experiment_name}
	pushd ${experiment_name}
	#JAX_DEBUG_NANS=True \
	#TODO(mkuchnik): Make sure these parameters are same for other calls
	python3 ${train_script} --data_dir=${data_dir} --val_data_dir=${val_data_dir} --output_dir=$output_dir \
		--epochs_per_loop=${epochs_per_loop} --xprof=False --cache=False --resnet_depth=${resnet_depth} --no_eval=False \
		--optimize_plumber_pipeline=False \
		--input_pipeline_default_prefetching=${input_pipeline_default_prefetching} \
		--input_pipeline_default_parallelism=${input_pipeline_default_parallelism} \
		--input_pipeline_threadpool_size=${input_pipeline_threadpool_size} \
		--optimizer=momentum \
		--learning_rate=0.1 \
		--momentum=0.9 \
		--num_epochs=${num_epochs} \
		--weight_decay=1e-4 \
		--mixup_alpha=${mixup_alpha} \
		--label_smoothing=${label_smoothing} \
		--cache=False \
		--optimize_plumber_pipeline=${optimize_plumber_pipeline} \
		--echoing_factor=${echo_factor} \
		--echoing_shuffle_buffer_size=${echo_shuffle_buffer_size} \
		--randaugment_num_layers=${randaugment_num_layers} \
		--randaugment_magnitude=${randaugment_magnitude} \
		--fake_model=${fake_model} \
		--force_cache_uncompressed=${force_cache_uncompressed} \
		--epochs_per_loop_scaler=${epochs_per_loop_scaler} \
		--no_eval=${no_eval} \
		2>&1 | tee log.txt
	popd
}

run_training_bench_heuristic_no_prefetch() {
	python3 train.py --data_dir=${data_dir} --val_data_dir=${val_data_dir} --output_dir=$output_dir \
		--epochs_per_loop=${epochs_per_loop} --xprof=False --cache=False --resnet_depth=${resnet_depth} --no_eval=True --input_pipeline_default_parallelism=64 \
		--optimize_plumber_pipeline=False \
		--input_pipeline_default_prefetching=0
}

# Synthetic

run_training_bench_synthetic() {
	python3 train.py --data_dir=${data_dir} --val_data_dir=${val_data_dir} --output_dir=$output_dir \
		--epochs_per_loop=${epochs_per_loop} --xprof=False --cache=False --resnet_depth=${resnet_depth} --no_eval=True --input_pipeline_default_parallelism=64 \
		--fake_data=True --local_batch_size=${batch_size}
}

run_training_bloated() {
	python3 train.py --data_dir=${data_dir} --val_data_dir=${val_data_dir} --output_dir=$output_dir \
		--epochs_per_loop=${epochs_per_loop} --xprof=True --cache=False --memory_bloat_percentage=0.75
}

run_benchmark_pipeline() {
	experiment_name=${experiment_prefix}HEURISTIC_TRAINING_benchmark_echo_${echo_factor}
	mkdir -p ${experiment_name}
	pushd ${experiment_name}
	python3 ${benchmark_script} --data_dir=${data_dir} --time_limit_s=${bench_time_limit_s} --cache=False \
		--mixup_alpha=${mixup_alpha} \
		--optimize_plumber_pipeline=${optimize_plumber_pipeline} \
		--input_pipeline_default_prefetching=${input_pipeline_default_prefetching} \
		--input_pipeline_default_parallelism=${input_pipeline_default_parallelism} \
		--input_pipeline_threadpool_size=${input_pipeline_threadpool_size} \
		--randaugment_num_layers=${randaugment_num_layers} \
		--randaugment_magnitude=${randaugment_magnitude} \
		--force_cache_uncompressed=${force_cache_uncompressed} \
		--echoing_factor=${echo_factor} \
		2>&1 | tee log.txt
	popd
}
