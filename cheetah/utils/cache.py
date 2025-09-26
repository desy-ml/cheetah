import functools

import torch


def cache_transfer_map(func):
    """
    Decorator to cache the transfer map of an accelerator element based on its
    defining features, the beam energy and the particle species.
    """
    from cheetah.accelerator.element import Element
    from cheetah.particles.species import Species

    @functools.wraps(func)
    def wrapper(self: Element, energy: torch.Tensor, species: Species) -> torch.Tensor:
        # Caching is not supported if any of input tensors require gradients
        if any(
            x.requires_grad
            for x in (energy, species.num_elementary_charges, species.mass_eV)
        ):
            return func(self, energy, species)

        if not hasattr(self, "_cache"):
            self._cache = {}
        if func.__name__ not in self._cache:
            self._cache[func.__name__] = {}
        cache = self._cache[func.__name__]

        # Build a validity key to check if element features have changed
        feature_validity_key = tuple()
        for feature_name in self.defining_features:
            feature = getattr(self, feature_name)
            if isinstance(feature, torch.Tensor):
                feature_validity_key += (
                    id(feature),
                    feature._version,
                    feature.requires_grad,
                )
            else:
                feature_validity_key += (feature,)

        # Recompute the transfer map if element features have changed
        if feature_validity_key != cache.get("feature_validity_key", None) or any(
            not (
                passed.dtype == cached.dtype
                and passed.device == cached.device
                and passed.requires_grad == cached.requires_grad
                and torch.equal(passed, cached)
            )
            for passed, cached in zip(
                (energy, species.num_elementary_charges, species.mass_eV),
                (
                    cache.get("energy"),
                    cache.get("num_elementary_charges"),
                    cache.get("mass_eV"),
                ),
            )
        ):
            cache["result"] = func(self, energy, species)

            cache["feature_validity_key"] = feature_validity_key
            cache["energy"] = energy.clone()
            cache["num_elementary_charges"] = species.num_elementary_charges.clone()
            cache["mass_eV"] = species.mass_eV.clone()

        return cache["result"]

    return wrapper
