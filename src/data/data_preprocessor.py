class DataPreprocessor:
    """This class will process tf.data.Dataset to
    add various pre-processing on top.
    Also finally make an initializable iterator with several
    data load options.

    Parameters
    ----------
    dataset: tf.Dataset
        Dataset to add pre-processes.
    num_parallel_calls: int
        Parameter for tf.Dataset.map.
        [default: 10]
    batch_size: int
        Make a batch of size of
        @p batch_size. [default: 32]
    shuffle_buffer_size: int
        If not None or False, shuffle datasets
        with @p shuffle_buffer_size.
        [default: 100]
    prefetch_buffer_size: int
        If not None or False, prefetch datasets
        with @p prefetch_buffer_size
        [default: 100]
    """

    def __init__(self,
                 dataset,
                 anchor_converter,
                 num_parallel_calls=10,
                 batch_size=32,
                 shuffle_buffer_size=100,
                 prefetch_buffer_size=1):
        self.dataset = dataset
        self.anchor_converter = anchor_converter
        self.num_parallel_calls = num_parallel_calls
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size

    def process_image(self, process_fn, **kwargs):
        """Add set of pre-processes on image using tf.Dataset.map.

        Parameters
        ----------
        process_fn: functional.
            A functional to process image data.
            The functional should accept @p image argument, and returns
            a tf.Tensor with the same shape to @p image.
        kwargs: dict
            A parameter dictionary which is necessary for the functionals.
        """

        def processor(x):
            x['image'] = process_fn(image=x['image'], **kwargs)
            return x

        self.dataset = self.dataset.map(lambda x: processor(x),
                                        self.num_parallel_calls)

    def process_label(self, process_fn, **kwargs):
        """Add set of pre-processes on label using tf.Dataset.map.

        Parameters
        ----------
        process_fn: functional.
            A functional to process label data.
            The functional should accept @p label argument, and returns
            a tf.Tensor with the same shape to @p label.
        kwargs: dict
            A parameter dictionary which is necessary for the functionals.
        """

        def processor(x):
            x['label'] = process_fn(label=x['label'], **kwargs)
            return x

        self.dataset = self.dataset.map(lambda x: processor(x),
                                        self.num_parallel_calls)

    def process_image_and_label(self, process_fn, **kwargs):
        """Add set of pre-processes on both image and label
        using tf.Dataset.map.

        Parameters
        ----------
        process_fn: functional.
            A functional to process both image and label data.
            The functional should accept @p image and @p label arguments,
            and returns two tf.Tensor with the same shape to @p image and @p label.
        kwargs: dict
            A parameter dictionary which is necessary for the functionals.
        """

        def processor(x):
            x['image'], x['label'] = process_fn(
                image=x['image'], label=x['label'], **kwargs)
            return x

        self.dataset = self.dataset.map(lambda x: processor(x),
                                        self.num_parallel_calls)

    def get_iterator(self):
        """Make one shot iterator with several data load options
        include shuffle, batching, and prefetching.

        Returns
        -------
        iterator: tf.data.Iterator
            The initializable iterator.
        """
        # Generate anchor targets before batching to align the number of boxes.
        self.process_label(self.anchor_converter.generate_anchor_targets)

        if self.shuffle_buffer_size not in (None, False):
            self.dataset = self.dataset.shuffle(self.shuffle_buffer_size)
        self.dataset = self.dataset.batch(self.batch_size)
        if self.prefetch_buffer_size not in (None, False):
            self.dataset = self.dataset.prefetch(self.prefetch_buffer_size)
        iterator = self.dataset.make_initializable_iterator()
        return iterator
