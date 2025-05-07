def squash_index_for_unavailable_dims(index: tuple, shape: tuple) -> tuple:
    """
    Helper function to create an index that works on only partially broadcasted tensors.

    You might need this when you vectorized one parameter of your model as `(n,)` and
    another as `(m, 1)`. Fully broadcasted, this would result in vector shapes of
    `(m, n)`. Some functions in Cheetah expect an index into the fully broadcasted
    vector shape. However, sometimes this index is passed onto a function that only
    handles a result from the model that was only affected by one of these
    vectorizations, e.g., the second one. This function lets you squash the vector index
    meant for a vectorised result of shape `(m, n)` into a vector index that works on
    a result with vector shape `(m, 1)` or `(n,)`.

    Example: The input vector shapes are (a) `(3,)` and (b) `(2, 1)`, resulting in a
    broadcasted shape of `(2, 3)`. A valid vector index for the fully broadcasted model
    would be `(1, 2)`. If the result is only affected by vectorisation (b) with shape
    `(2, 1)`, this function would squash the second dimension of the index to `0`,
    resulting in `(1, 0)`. Similarly, if the result is only affected by vectorisation
    (a) with shape `(3,)`, this function would remove the first dimension of the index,
    resulting in `(2,)`.

    :param index: The index to squash.
    :param shape: The shape of the result that was only affected by part of the
        vectorisations.
    :return: The squashed index that is valid for indexing into a tensor of `shape`.
    """
    length_squashed = index[-len(shape) :]
    unavailable_dims_squashed = tuple(
        0 if s == 1 else i for i, s in zip(length_squashed, shape)
    )

    return unavailable_dims_squashed
