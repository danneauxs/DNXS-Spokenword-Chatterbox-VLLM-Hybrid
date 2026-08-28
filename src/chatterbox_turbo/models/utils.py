class AttrDict(dict):
    """A dictionary subclass that allows attribute-style access to its keys."""
    def __init__(self, *args, **kwargs):
        """Converts an instance to a dictionary where attributes can be accessed like keys.
        Args:
        *args: Positional arguments passed to the superclass constructor.
        **kwargs: Keyword arguments passed to the superclass constructor.
        Returns:
        None. The method modifies the instance in place.
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
