import tensorflow as tf

from efficientnet_preprocessing import autoaugment


def apply_randaugment_to_image(image, randaug_num_layers, randaug_magnitude,
                               dtype):
    input_image_type = image.dtype
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, dtype=tf.uint8)

    image = autoaugment.distort_image_with_randaugment(
        image, randaug_num_layers, randaug_magnitude)

    image = tf.cast(image, dtype=input_image_type)
    image = tf.image.convert_image_dtype(
        image, dtype=dtype)
    return image


def apply_autoaugment_to_image(image, dtype):
    input_image_type = image.dtype
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, dtype=tf.uint8)

    image = autoaugment.distort_image_with_autoaugment(image, 'v0')

    image = tf.cast(image, dtype=input_image_type)
    image = tf.image.convert_image_dtype(
        image, dtype=dtype)
    return image
