import tensorflow.compat.v1 as tf 

def mixup(batch_size, alpha, images, labels):
    """Applies Mixup regularization to a batch of images and labels.

    [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
      Mixup: Beyond Empirical Risk Minimization.
      ICLR'18, https://arxiv.org/abs/1710.09412

    Arguments:
      batch_size: The input batch size for images and labels.
      alpha: Float that controls the strength of Mixup regularization.
      images: A batch of images of shape [batch_size, ...]
      labels: A batch of labels of shape [batch_size, num_classes]

    Returns:
      A tuple of (images, labels) with the same dimensions as the input with
      Mixup regularization applied.
    """
    mix_weight = tf.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
    images_mix_weight = tf.cast(
        tf.reshape(mix_weight, [batch_size, 1, 1, 1]), images.dtype)
    # Mixup on a single batch is implemented by taking a weighted sum with the
    # same batch in reverse.
    images_mix = (
        images * images_mix_weight + images[::-1] * (1. - images_mix_weight))
    labels_float = tf.cast(labels, dtype=tf.float32)
    labels_mix = labels_float * mix_weight + labels_float[::-1] * (1. - mix_weight)
    return images_mix, labels_mix

def fake_mixup(batch_size, alpha, images, labels):
    """
    Does nothing
    """
    return images, labels
