from __future__ import annotations
from typing import Any, Iterable, TypeVar, Type, Union, List
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

def coerce_to_model(obj: Any, model: Type[T]) -> T:
    """
    Convert a Python object (usually a dict) into a Pydantic model instance.
    - Uses Pydantic v2's model_validate; for v1 use model.parse_obj.
    - Raises ValidationError on bad inputs.
    """
    # Pydantic v2:
    return model.model_validate(obj)  # type: ignore[attr-defined]

    # If you're on Pydantic v1, replace with:
    # return model.parse_obj(obj)

def coerce_each(items: Iterable[Any], model: Type[T]) -> List[T]:
    """Convert an iterable of dict-like items into a list of model instances."""
    out: List[T] = []
    for ix, item in enumerate(items, 1):
        try:
            out.append(coerce_to_model(item, model))
        except ValidationError as e:
            # You can aggregate or log; here we raise with context
            raise ValidationError(
                e.errors(), model=model
            ) from RuntimeError(f"Item #{ix} failed validation")
    return out
