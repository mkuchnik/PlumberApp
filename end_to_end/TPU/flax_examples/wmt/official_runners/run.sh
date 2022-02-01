TFDS_DATA_DIR="gs://mkuchnik_data_eu_west4/data/tfds"

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

# NOTE default is config.per_device_batch_size = 32
BATCH_SIZE=128

export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_VLOG_LEVEL=0


run_experiment() {
	TFDS_DATA_DIR=$TFDS_DATA_DIR python3 main.py --workdir=$PWD/logs/wmt_256 \
            --config="configs/plumber_experiment_config.py" \
	    --config.per_device_batch_size=${BATCH_SIZE} \
	    2>&1 | tee run_log.txt
}

kill_python_programs
run_experiment
