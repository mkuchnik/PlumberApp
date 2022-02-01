import tracing_util

def apply_tracing_to_iterator(iterator):
    callback = tracing_util.CallbackConstructor()
    traced_iter = tracing_util.TracedIterator(iterator, lambda x: callback.callback_fn(x))
    return traced_iter, callback
