"""Natural-language steering of the desk page."""

from gendesk.steering.instructions import (
    Instruction,
    apply_instruction,
    parse_instruction,
)

__all__ = ["Instruction", "apply_instruction", "parse_instruction"]
