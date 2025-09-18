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

        # Check if any of the inputs or defining features have changed by building a
        # validity key
        new_validity_key_arg_part = tuple(
            (arg.tolist(), arg.device, arg.dtype, arg.requires_grad)
            for arg in (energy, species.num_elementary_charges, species.mass_eV)
        )
        new_validity_key_feature_part = tuple()
        for feature_name in self.defining_features:
            feature = getattr(self, feature_name)
            if isinstance(feature, torch.Tensor):
                new_validity_key_feature_part += (
                    id(feature),
                    feature._version,
                    feature.requires_grad,
                )
            else:
                new_validity_key_feature_part += (feature,)
        new_validity_key = new_validity_key_arg_part + new_validity_key_feature_part

        if not hasattr(self, "_cache"):
            self._cache = {}
        cache = self._cache
        validity_key_dict_key = f"{func.__name__}_validity_key"
        result_dict_key = f"{func.__name__}_result"

        # Recompute the transfer map if the validity keys do not match
        if new_validity_key != cache.get(validity_key_dict_key, None):
            result = func(self, energy, species)

            cache[result_dict_key] = result
            cache[validity_key_dict_key] = new_validity_key
        else:
            result = cache[result_dict_key]

        return result

    return wrapper
