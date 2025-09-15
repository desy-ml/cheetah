import torch


def cache_transfer_map(func):
    """
    Decorator to cache the transfer map of an accelerator element based on its
    defining features, the beam energy and the particle species.
    """
    from cheetah.accelerator.element import Element
    from cheetah.particles.species import Species

    def wrapper(self: Element, energy: torch.Tensor, species: Species) -> torch.Tensor:
        # Caching not supported if any of input tensors require gradients
        if any(
            x.requires_grad
            for x in (energy, species.num_elementary_charges, species.mass_eV)
        ):
            return func(self, energy, species)

        cached_transfer_map_attr_name = f"_cached_{func.__name__}_result"

        # Recompute and cache if any of the inputs or defining features have changed
        new_cache_validity_key = tuple(
            (arg.item(), arg.requires_grad)
            for arg in (energy, species.num_elementary_charges, species.mass_eV)
        ) + tuple(
            (
                feature
                if not isinstance(feature, torch.Tensor)
                else (id(feature), feature._version, feature.requires_grad)
            )
            for feature in self.defining_features
        )
        saved_cache_validity_key_attr_name = f"_cached_{func.__name__}_validity_key"
        saved_cache_validity_key = getattr(
            self, saved_cache_validity_key_attr_name, None
        )
        if new_cache_validity_key != saved_cache_validity_key:
            # Recompute the transfer map
            result = func(self, energy, species)

            # Ensure that a buffer is registered to hold the cached result
            if not hasattr(self, cached_transfer_map_attr_name):
                self.register_buffer(cached_transfer_map_attr_name, result)
            else:
                setattr(self, cached_transfer_map_attr_name, result)

            # Save the new validity key for later cache checks
            setattr(self, saved_cache_validity_key_attr_name, new_cache_validity_key)

        return getattr(self, cached_transfer_map_attr_name)

    return wrapper
