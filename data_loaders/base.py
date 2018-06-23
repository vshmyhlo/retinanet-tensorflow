class Base(object):
    @property
    def class_names(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
