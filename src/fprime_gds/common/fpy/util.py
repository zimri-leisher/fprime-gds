from typing import Union, get_args, get_origin

# In Python 3.10+, the `|` operator creates a `types.UnionType`.
# We need to handle this for forward compatibility, but it won't exist in 3.9.
try:
    from types import UnionType

    UNION_TYPES = (Union, UnionType)
except ImportError:
    UNION_TYPES = (Union,)


def is_instance_compat(obj, cls):
    """
    A wrapper for isinstance() that correctly handles Union types in Python 3.9+.

    Args:
        obj: The object to check.
        cls: The class, tuple of classes, or Union type to check against.

    Returns:
        True if the object is an instance of the class or any type in the Union.
    """
    origin = get_origin(cls)
    if origin in UNION_TYPES:
        # It's a Union type, so get its arguments.
        # e.g., get_args(Union[int, str]) returns (int, str)
        return isinstance(obj, get_args(cls))

    # It's not a Union, so it's a regular type (like int) or a
    # tuple of types ((int, str)), which isinstance handles natively.
    return isinstance(obj, cls)