def squash_index_for_unavailable_dims(index: tuple, shape: tuple) -> tuple:
    """
    Helper function to create an index that works on only partially broadcasted tensors.

    You might need this, when you vectorised one parameter of your model as `(n,)` and
    another as `(m, 1)`. Fully broadcasted, this would result in vector shapes of
    `(m, n)`. Some functions in Cheetah expect an index into the fully broadcasted
    vector shape. However, sometimes this index is passed onto a function that only
    handles a result from the model that was only affected by one of these
    vectorisations, e.g. the second one. This function let's you squash the vector index
    meant for a vectorised result of shape `(m, n)` into a vector index that works on
    a result with vector shape `(m, 1)`.

    Example: The input vector shapes are `(3,)` and `(2, 1)`, resulting in a broadcasted
    shape of `(3, 2)`. A valid vector index for fully-broadcasted model would be
    `(0, 2)`, which is not valid for results only affected by the second vectorisation.
    This function would then squash the second dimension of the index, where the result
    is not vectorised, to `0`, resulting a valid index of `(0, 0)`.

    :param index: The index to squash.
    :param shape: The shape of the result that was only affected part of the
        vectorisations.
    :return: The squashed index that is valid indexing into a tensor of `shape`.
    """
    length_squashed_index = index[-len(shape) :]
    unavailable_dims_squashed = tuple(
        0 if s == 1 else i for i, s in zip(length_squashed_index, shape)
    )

    return unavailable_dims_squashed
