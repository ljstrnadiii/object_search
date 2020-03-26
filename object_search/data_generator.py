from multiprocessing import Pool, Process
import numpy as np
from math import ceil

class BackgroundGenerator():
    """Spawns processes to prepare an iterator for background dataset
    generation. This is useful if you do not want to confine yourself
    to the tensorflow dataset api.

    If utilized propoerly, we can maximize cpu and gpu resources.

    Example:
    ```
    bgdg = BackgroundGenerator(proc=read_transform,
                               inputs=inputs,
                               batch_size=100,
                               n_parallel=4,
                               chunksize=5)

    for data, inputs in bgdb:
        results = model(data)
    ```

    Attributes:
        output_size: tuple
            the shape of the output in order to construct matrix to hold
            batches
    """
    def __init__(self, proc, inputs, batch_size, n_parallel, chunksize):
        """Initialize object, compute the expected output size

        Args:
            proc: object
                callable function that computes on elements on inputs
            inputs: list
                list of inputs for the proc function. e.g. list of paths
            n_parallel: int
                The number of processes to use for the background data
                generator
            batch_size: int
                the batch size to be returned
            chunksize: int
                The approximate number of elements to be computed in each
                process.
            """
        # inputs
        self.proc = proc
        self.inputs = inputs
        self.n_parallel = n_parallel
        self.chunksize = chunksize
        self.batch_size = batch_size
        # prep
        self.output_size = self.get_output_size()
        self.start_processing()

    def get_output_size(self):
        """Compute the output size for the batch placeholder"""
        output = self.proc(self.inputs[0])
        assert np.all( x in output.keys() for x in ['x','input']),\
                "proc must return dictionary with 'x' and 'input' as keys."
        return output['x'].shape

    def start_processing(self):
        """Initiates the background process of proc on inputs"""
        self.pool = Pool(self.n_parallel)
        self.processed = self.pool.imap_unordered(self.proc,
                                             self.inputs,
                                             chunksize=self.chunksize)

    def next(self):
        """Used by __iter__ to construct the iterator

        Returns:
            batch: np.ndarray
                The batch
            inputs: list
                The corresponding inputs for the batch in order
        """
        inputs = []
        batch = np.zeros((self.batch_size, *self.output_size))
        try:
            for i in range(self.batch_size):
                data = next(self.processed)
                x = data['x']
                ins = data['input']
                batch[i] = x
                inputs.append(ins)
        except StopIteration:
            i -= 1
            pass

        if len(inputs)==0:
            raise StopIteration

        return batch[:i+1], inputs

    def __iter__(self):
        """The function responsible for making this class iterable. Refer
        to self.next() for the return type."""
        for i in range(ceil(len(self.inputs)/self.batch_size)):
            yield self.next()

    def __del__(self):
        self.pool.close()
        self.pool.join()
