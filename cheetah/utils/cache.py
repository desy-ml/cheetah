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

        # Recompute and cache if any of the inputs or defining features have changed
        if not all(
            _is_tensor_arg_cache_hit(self, arg, arg_name, func.__name__)
            for arg, arg_name in (
                (energy, "energy"),
                (species.num_elementary_charges, "num_elementary_charges"),
                (species.mass_eV, "mass_eV"),
            )
        ) or not all(
            _is_defining_feature_cache_hit(self, feature_name, func.__name__)
            for feature_name in self.defining_features
        ):
            # Recompute
            result = func(self, energy, species)

            # Make sure tensor cache variables are registered
            if not hasattr(self, f"_cached_{func.__name__}_result"):
                self.register_buffer(
                    f"_cached_{func.__name__}_result", None, persistent=False
                )
            for arg_name in ["energy", "num_elementary_charges", "mass_eV"]:
                if not hasattr(self, f"_cached_{func.__name__}_{arg_name}_value"):
                    self.register_buffer(
                        f"_cached_{func.__name__}_{arg_name}_value",
                        None,
                        persistent=False,
                    )

            # Update cache
            setattr(self, f"_cached_{func.__name__}_result", result)

            setattr(self, f"_cached_{func.__name__}_energy_value", energy.clone())
            setattr(
                self,
                f"_cached_{func.__name__}_num_elementary_charges_value",
                species.num_elementary_charges.clone(),
            )
            setattr(
                self, f"_cached_{func.__name__}_mass_eV_value", species.mass_eV.clone()
            )
            for feature in self.defining_features:
                if isinstance(getattr(self, feature), torch.Tensor):
                    setattr(
                        self,
                        f"_cached_{func.__name__}_{feature}_id",
                        id(getattr(self, feature)),
                    )
                    setattr(
                        self,
                        f"_cached_{func.__name__}_{feature}_version",
                        getattr(self, feature)._version,
                    )
                    setattr(
                        self,
                        f"_cached_{func.__name__}_{feature}_requires_grad",
                        getattr(self, feature).requires_grad,
                    )
                else:
                    setattr(
                        self,
                        f"_cached_{func.__name__}_{feature}_hash",
                        hash(getattr(self, feature)),
                    )

        # Return cached result
        return getattr(self, f"_cached_{func.__name__}_result")

    return wrapper


def _is_tensor_arg_cache_hit(
    self, arg: torch.Tensor, arg_name: str, func_name: str
) -> bool:
    """
    Check if the cache for a tensor-valued argument to the function `func_name` is still
    valid. This is the case if the argument and cached tensor have the same value and
    gradient function.
    """
    cached_tensor = getattr(self, f"_cached_{func_name}_{arg_name}_value", None)

    return (
        cached_tensor is not None
        and torch.equal(arg, cached_tensor)
        and arg.grad_fn == cached_tensor.grad_fn
    )


def _is_defining_feature_cache_hit(self, feature_name: str, func_name: str) -> bool:
    """
    Check if the cache for a defining feature of `self` is still valid. Distinguishes
    between tensor and non-tensor features. Tensor features are considered a hit if
    their memory ID, version and gradient requirement are the same. Non-tensor features
    are considered a hit if their hash value is the same.
    """
    feature = getattr(self, feature_name)

    if isinstance(feature, torch.Tensor):
        return (
            id(feature) == getattr(self, f"_cached_{func_name}_{feature_name}_id", None)
            and feature._version
            == getattr(self, f"_cached_{func_name}_{feature_name}_version", None)
            and feature.requires_grad
            == getattr(self, f"_cached_{func_name}_{feature_name}_requires_grad", None)
        )
    else:
        return hash(feature) == getattr(
            self, f"_cached_{func_name}_{feature_name}_hash", None
        )
