class DeviceError(Exception):
    """
    Used to create an exception, in case the device used for the beam
    and the elements are different.
    """

    def __init__(self):
        super().__init__(
            "Warning! The device used for calculating the elements is not the same, "
            "as the device used to calculate the Beam."
        )
