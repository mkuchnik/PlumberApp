"""Utilities to use tf.data service."""

import tensorflow as tf

def apply_service_to_dataset(dataset, service_ip, job_name=None):
    """Applies tfdata service to the dataset. Specify IP with arg, and
    optionally the job name"""
    if job_name:
        job_name = "my_tfdata_service"
    dataset = dataset.apply(
        tf.data.experimental.service.distribute(
            processing_mode='parallel_epochs',
            service=service_ip,
            job_name=job_name))
    return dataset
