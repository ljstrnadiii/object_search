from object_search.data_processors import preprocess, postprocess
from object_search.data_processors import preprocess, postprocess


class TestPipeline:
    """Tests from file reading to prep for inference."""

    def test_load_generator(self, write_images_to_file):
        paths, imgs = write_images_to_file
        #dataset = BackgroundGenerator(
        #    proc=proc,
        #    inputs=inputs,
        #    n_parallel=2,
        #    batch_size=16,
        #    chunksize=2)
        #return dataset

        assert type(paths) is list

    def test_image_file_read(self):
        # read file test shape
        pass

    def test_generator(self, write_images_to_file):
        paths, imgs = write_images_to_file
        # given test input, check for iterator
        pass

    def test_generator_type(self):
        pass
