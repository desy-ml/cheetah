class UniqueNameGenerator:
    """Generates a unique name given a prefix."""

    def __init__(self, prefix):
        self._prefix = prefix
        self._counter = 0

    def __call__(self):
        name = f"{self._prefix}_{self._counter}"
        self._counter += 1
        return name
