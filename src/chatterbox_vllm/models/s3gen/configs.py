class AttrDict(dict):
    """Defines a class that behaves like a dictionary but allows access to keys as attributes."""
    def __init__(self, *args, **kwargs):
        """Initializes an AttrDict object with dictionary attributes accessible as instance properties."""
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

CFM_PARAMS = AttrDict({
    "sigma_min": 1e-06,
    "solver": "euler",
    "t_scheduler": "cosine",
    "training_cfg_rate": 0.2,
    "inference_cfg_rate": 0.7,
    "reg_loss_type": "l1"
})