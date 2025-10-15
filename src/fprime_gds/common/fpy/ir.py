from dataclasses import dataclass, field
import inspect
import typing

from fprime_gds.common.fpy.backend_types import BackendState
from fprime_gds.common.fpy.bytecode.directives import (
    BinaryStackOp,
    Directive,
    UnaryStackOp,
)
from fprime.common.models.serialize.type_base import BaseType as FppValue
from fprime_gds.common.fpy.frontend_types import is_instance_compat

# a value of type FppType is a Python `type` object representing
# the type of an Fprime value
FppType = type[FppValue]


@dataclass
class Ir:
    id: int = field(init=False, repr=False, default=None)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, value):
        if not isinstance(value, Ir):
            return False
        assert self.id is not None
        return self.id == value.id


@dataclass
class IrStmt(Ir):
    pass


@dataclass
class IrInstruction(IrStmt):
    # instruction generates strictly one directive
    pass


@dataclass
class IrDirective(IrInstruction):
    dir: Directive


@dataclass
class IrGoto(IrInstruction):
    label: str


@dataclass
class IrIf(IrInstruction):
    goto_false_label: str


@dataclass
class IrBasicBlock:
    name: str
    stmts: list[IrStmt]
    successors: list["IrBasicBlock"]
    predecessors: list["IrBasicBlock"]


@dataclass
class IrFunction:
    args: dict[str, FppType]
    return_type: FppType
    blocks: list[IrBasicBlock]
    # entry is first block


@dataclass
class IrModule:
    funcs: list[IrFunction]


class IrVisitor:

    def _find_custom_visit_func(self, stmt: IrStmt):
        for name, func in inspect.getmembers(type(self), inspect.isfunction):
            if not name.startswith("visit"):
                # not a visitor
                continue
            signature = inspect.signature(func)
            params = list(signature.parameters.values())
            assert len(params) == 6
            assert params[1].annotation is not None
            annotations = typing.get_type_hints(func)
            param_type = annotations[params[1].name]
            if is_instance_compat(stmt, param_type):
                return getattr(self, name)
        return self.visit_stmt_default

    def _visit_stmt(
        self,
        stmt: IrStmt,
        block: IrBasicBlock,
        func: IrFunction,
        mod: IrModule,
        state: BackendState,
    ):
        visit_func = self._find_custom_visit_func(stmt)
        visit_func(stmt, block, func, mod, state)

    def visit_stmt_default(
        self,
        stmt: IrStmt,
        block: IrBasicBlock,
        func: IrFunction,
        mod: IrModule,
        state: BackendState,
    ):
        pass

    def visit_basic_block(
        self, block: IrBasicBlock, func: IrFunction, mod: IrModule, state: BackendState
    ):
        pass

    def visit_function(self, func: IrFunction, mod: IrModule, state: BackendState):
        pass

    def visit_module(self, mod: IrModule, state: BackendState):
        pass

    def run(self, mod: IrModule, state: BackendState):
        """runs the visitor, starting at the module, descending depth-first"""

        for func in mod.funcs:
            for block in func.blocks:
                for stmt in block.stmts:
                    self._visit_stmt(stmt, block, func, mod, state)
                    if len(state.errors) != 0:
                        return
                self.visit_basic_block(block, func, mod, state)
                if len(state.errors) != 0:
                    return
            self.visit_function(func, mod, state)
            if len(state.errors) != 0:
                return

        self.visit_module(mod, state)
        if len(state.errors) != 0:
            return
