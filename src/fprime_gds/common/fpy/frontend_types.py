from __future__ import annotations
from abc import ABC
import inspect
from dataclasses import astuple, dataclass, field, fields
from pathlib import Path
import typing
from typing import Union, get_args, get_origin
from fprime_gds.common.fpy.error import FrontendError
from fprime_gds.common.fpy.ir import IrBasicBlock, IrInstruction
from typing import Iterable, Union, get_args, get_origin
import zlib


# In Python 3.10+, the `|` operator creates a `types.UnionType`.
# We need to handle this for forward compatibility, but it won't exist in 3.9.
try:
    from types import UnionType

    UNION_TYPES = (Union, UnionType)
except ImportError:
    UNION_TYPES = (Union,)

from fprime_gds.common.fpy.bytecode.directives import (
    StackOpDirective,
    FloatLogDirective,
    Directive,
    ExitDirective,
    WaitAbsDirective,
    WaitRelDirective,
)
from fprime_gds.common.templates.ch_template import ChTemplate
from fprime_gds.common.templates.cmd_template import CmdTemplate
from fprime_gds.common.templates.prm_template import PrmTemplate
from fprime.common.models.serialize.time_type import TimeType
from fprime.common.models.serialize.serializable_type import SerializableType
from fprime.common.models.serialize.array_type import ArrayType
from fprime.common.models.serialize.numerical_types import (
    U32Type,
    U16Type,
    U64Type,
    U8Type,
    I16Type,
    I32Type,
    I64Type,
    I8Type,
    F32Type,
    F64Type,
    IntegerType,
)
from fprime.common.models.serialize.string_type import StringType
from fprime.common.models.serialize.bool_type import BoolType
from fprime_gds.common.fpy.syntax import (
    AstBody,
    AstExpr,
    AstFor,
    AstOp,
    AstReference,
    Ast,
    AstAssign,
    AstScopedBody,
    AstStmt,
    AstVar,
)
from fprime.common.models.serialize.type_base import BaseType as FppValue

MAX_DIRECTIVES_COUNT = 1024
MAX_DIRECTIVE_SIZE = 2048
MAX_STACK_SIZE = 1024

COMPILER_MAX_STRING_SIZE = 128


# this is the "internal" integer type that integer literals have by
# default. it is arbitrary precision
class InternalIntType(IntegerType):
    @classmethod
    def range(cls):
        raise NotImplementedError()

    @staticmethod
    def get_serialize_format():
        raise NotImplementedError()

    @classmethod
    def get_bits(cls):
        raise NotImplementedError()

    @classmethod
    def validate(cls, val):
        if not isinstance(val, int):
            raise RuntimeError()


InternalStringType = StringType.construct_type("InternalStringType", None)


SPECIFIC_NUMERIC_TYPES = (
    U32Type,
    U16Type,
    U64Type,
    U8Type,
    I16Type,
    I32Type,
    I64Type,
    I8Type,
    F32Type,
    F64Type,
)
SPECIFIC_INTEGER_TYPES = (
    U32Type,
    U16Type,
    U64Type,
    U8Type,
    I16Type,
    I32Type,
    I64Type,
    I8Type,
)
SIGNED_INTEGER_TYPES = (
    I16Type,
    I32Type,
    I64Type,
    I8Type,
)
UNSIGNED_INTEGER_TYPES = (
    U32Type,
    U16Type,
    U64Type,
    U8Type,
)
SPECIFIC_FLOAT_TYPES = (
    F32Type,
    F64Type,
)

ArrayIndexType = U64Type


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


# a value of type FppType is a Python `type` object representing
# the type of an Fprime value
FppType = type[FppValue]


class NothingValue(ABC):
    """a type which has no valid values in fprime. used to denote
    a function which doesn't return a value"""

    @classmethod
    def __subclasscheck__(cls, subclass):
        return False


# the `type` object representing the NothingValue class
NothingType = type[NothingValue]


@dataclass
class FpyCallable:
    return_type: FppType | NothingType
    args: list[tuple[str, FppType]]


@dataclass
class FpyCmd(FpyCallable):
    cmd: CmdTemplate


@dataclass
class FpyMacro(FpyCallable):
    dir: type[Directive]
    """a function which instantiates the macro given the argument exprs"""


MACROS: dict[str, FpyMacro] = {
    "sleep": FpyMacro(
        NothingValue,
        [
            (
                "seconds",
                U32Type,
            ),
            ("microseconds", U32Type),
        ],
        WaitRelDirective,
    ),
    "sleep_until": FpyMacro(
        NothingValue, [("wakeup_time", TimeType)], WaitAbsDirective
    ),
    "exit": FpyMacro(NothingValue, [("exit_code", U8Type)], ExitDirective),
    "log": FpyMacro(F64Type, [("operand", F64Type)], FloatLogDirective),
}


@dataclass
class FpyTypeCtor(FpyCallable):
    type: FppType


@dataclass
class FieldReference:
    """a reference to a member/element of an fprime struct/array type"""

    parent_expr: AstExpr
    """the complete qualifier"""
    base_ref: FpyReference
    """the base ref, up through all the layers of field refs"""
    type: FppType
    """the fprime type of this reference"""
    is_struct_member: bool = False
    """True if this is a struct member reference"""
    is_array_element: bool = False
    """True if this is an array element reference"""
    base_offset: int = None
    """the constant offset in the base ref type, or None if unknown at compile time"""
    local_offset: int = None
    """the constant offset in the parent type at which to find this field
    or None if unknown at compile time"""
    name: str = None
    """the name of the field, if applicable"""
    idx_expr: AstExpr = None
    """the expression that evaluates to the index in the parent array of the field, if applicable"""


# named variables can be tlm chans, prms, callables, or directly referenced consts (usually enums)
@dataclass
class FpyVariable:
    """a mutable, typed value referenced by an unqualified name"""

    type_ref: AstExpr
    """the expression denoting the var's type"""
    declaration: AstAssign
    """the node where this var is declared"""
    type: FppType | None = None
    """the resolved type of the variable. None if type unsure at the moment"""
    lvar_offset: int | None = None
    """the offset in the lvar array where this var is stored"""

@dataclass
class ForLoopAnalysis:
    loop_var: FpyVariable
    loop_condition_lt_dir: type[StackOpDirective]
    

    for_loop_upper_bound_variables: dict[AstFor, FpyVariable] = field(
        default_factory=dict, repr=False
    )
    for_loop_comparison_directives: dict[AstFor, type[StackOpDirective]] = field(
        default_factory=dict, repr=False
    )
    for_loop_increment_directives: dict[AstFor, type[StackOpDirective]] = field(
        default_factory=dict, repr=False
    )
    for_loop_intermediate_type: dict[AstFor, FppTypeClass] = field(
        default_factory=dict, repr=False
    )

# a scope
next_scope_id = 0


class FpyScope(dict):
    def __init__(self):
        global next_scope_id
        self.id = next_scope_id
        next_scope_id += 1

    def __getitem__(self, key: str) -> FpyReference:
        return super().__getitem__(key)

    def get(self, key) -> FpyReference | None:
        return super().get(key, None)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, value):
        return isinstance(value, FpyScope) and value.id == self.id


def create_scope(
    references: dict[str, "FpyReference"],
) -> FpyScope:
    """from a flat dict of strs to references, creates a hierarchical, scoped
    dict. no two leaf nodes may have the same name"""

    base = FpyScope()

    for fqn, ref in references.items():
        names_strs = fqn.split(".")

        ns = base
        while len(names_strs) > 1:
            existing_child = ns.get(names_strs[0])
            if existing_child is None:
                # this scope is not defined atm
                existing_child = {}
                ns[names_strs[0]] = existing_child

            if not isinstance(existing_child, dict):
                # something else already has this name
                print(
                    f"WARNING: {fqn} is already defined as {existing_child}, tried to redefine it as {ref}"
                )
                break

            ns = existing_child
            names_strs = names_strs[1:]

        if len(names_strs) != 1:
            # broke early. skip this loop
            continue

        # okay, now ns is the complete scope of the attribute
        # i.e. everything up until the last '.'
        name = names_strs[0]

        existing_child = ns.get(name)

        if existing_child is not None:
            # uh oh, something already had this name with a diff value
            print(
                f"WARNING: {fqn} is already defined as {existing_child}, tried to redefine it as {ref}"
            )
            continue

        ns[name] = ref

    return base


def union_scope(lhs: FpyScope, rhs: FpyScope) -> FpyScope:
    """returns the two scopes, joined into one. if there is a conflict, chooses lhs over rhs"""
    lhs_keys = set(lhs.keys())
    rhs_keys = set(rhs.keys())
    common_keys = lhs_keys.intersection(rhs_keys)

    only_lhs_keys = lhs_keys.difference(common_keys)
    only_rhs_keys = rhs_keys.difference(common_keys)

    new = FpyScope()

    for key in common_keys:
        if not isinstance(lhs[key], dict) or not isinstance(rhs[key], dict):
            # cannot be merged cleanly. one of the two is not a scope
            print(f"WARNING: {key} is defined as {lhs[key]}, ignoring {rhs[key]}")
            new[key] = lhs[key]
            continue

        new[key] = union_scope(lhs[key], rhs[key])

    for key in only_lhs_keys:
        new[key] = lhs[key]
    for key in only_rhs_keys:
        new[key] = rhs[key]

    return new


FpyReference = typing.Union[
    ChTemplate,
    PrmTemplate,
    FppValue,
    FpyCallable,
    FppType,
    FpyVariable,
    FieldReference,
    dict,  # dict of FpyReference
]
"""some named concept in fpy"""


def get_ref_fpp_type_class(ref: FpyReference) -> FppType:
    """returns the fprime type of the ref, if it were to be evaluated as an expression"""
    if isinstance(ref, ChTemplate):
        result_type = ref.ch_type_obj
    elif isinstance(ref, PrmTemplate):
        result_type = ref.prm_type_obj
    elif isinstance(ref, FppValue):
        # constant value
        result_type = type(ref)
    elif isinstance(ref, FpyCallable):
        # a reference to a callable isn't a type in and of itself
        # it has a return type but you have to call it (with an AstFuncCall)
        # consider making a separate "reference" type
        result_type = NothingValue
    elif isinstance(ref, FpyVariable):
        result_type = ref.type
    elif isinstance(ref, type):
        # a reference to a type doesn't have a value, and so doesn't have a type,
        # in and of itself. if this were a function call to the type's ctor then
        # it would have a value and thus a type
        result_type = NothingValue
    elif isinstance(ref, FieldReference):
        result_type = ref.type
    elif isinstance(ref, dict):
        # reference to a scope. scopes don't have values
        result_type = NothingValue
    else:
        assert False, ref

    return result_type


def resolve_var(node: Ast, name: str, state: FrontendState) -> FpyVariable:
    # check this scope and all parent scopes
    local_scope = state.local_scopes[node]
    resolved = None
    while local_scope is not None and resolved is None:
        resolved = local_scope.get(name)
        local_scope = state.scope_parents[local_scope]

    return resolved


@dataclass
class FrontendState:
    """a collection of input, internal and output state variables and maps"""

    types: FpyScope
    """a scope whose leaf nodes are subclasses of BaseType"""
    callables: FpyScope
    """a scope whose leaf nodes are FpyCallable instances"""
    tlms: FpyScope
    """a scope whose leaf nodes are ChTemplates"""
    prms: FpyScope
    """a scope whose leaf nodes are PrmTemplates"""
    consts: FpyScope
    """a scope whose leaf nodes are FpyVariables"""
    runtime_values: FpyScope = None
    """a scope whose leaf nodes are tlms/prms/consts, all of which
    have some value at runtime."""

    def __post_init__(self):
        self.runtime_values = union_scope(
            self.tlms,
            union_scope(self.prms, self.consts),
        )

    root: AstScopedBody = None
    scope_parents: dict[AstScopedBody, AstScopedBody | None] = field(
        default_factory=dict, repr=False
    )
    body_scopes: dict[AstScopedBody, FpyScope] = field(default_factory=dict, repr=False)
    local_scopes: dict[Ast, FpyScope] = field(default_factory=dict, repr=False)
    for_loop_variables: dict[AstFor, FpyVariable] = field(
        default_factory=dict, repr=False
    )
    for_loop_upper_bound_variables: dict[AstFor, FpyVariable] = field(
        default_factory=dict, repr=False
    )
    for_loop_comparison_directives: dict[AstFor, type[StackOpDirective]] = field(
        default_factory=dict, repr=False
    )
    for_loop_increment_directives: dict[AstFor, type[StackOpDirective]] = field(
        default_factory=dict, repr=False
    )
    for_loop_intermediate_type: dict[AstFor, FppType] = field(
        default_factory=dict, repr=False
    )

    resolved_references: dict[AstReference, FpyReference] = field(
        default_factory=dict, repr=False
    )
    """reference to its singular resolution"""

    expr_unconverted_types: dict[AstExpr, FppType | NothingType] = field(
        default_factory=dict
    )
    """expr to its fprime type, before type conversions are applied"""

    stack_op_directives: dict[AstOp, type[StackOpDirective]] = field(
        default_factory=dict
    )
    """some stack operation to which directive will be emitted for it"""

    expr_converted_types: dict[AstExpr, FppType] = field(default_factory=dict)
    """expr to fprime type it will end up being on the stack after type conversions"""

    expr_converted_values: dict[AstExpr, FppValue | NothingValue | None] = field(
        default_factory=dict
    )
    """expr to the fprime value it will end up being on the stack after type conversions.
    None if unsure at compile time"""

    directives: dict[Ast, list[Directive] | None] = field(default_factory=dict)
    """a list of code generated by each node, or None/empty list if no directives"""

    stmt_directives: dict[AstStmt, list[Directive] | None] = field(default_factory=dict)

    basic_blocks: dict[Union[AstScopedBody, AstBody], IrBasicBlock] = field(default_factory=dict)

    node_dir_counts: dict[Ast, int] = field(default_factory=dict)
    """node to the number of directives generated by it"""

    lvar_array_size_bytes: int = 0
    """the size in bytes of the lvar array"""

    start_line_idx: dict[Ast, int] = field(default_factory=dict)
    """the line index at which each node's directives will be included in the output"""

    errors: list[FrontendError] = field(default_factory=list)
    """a list of all compile exceptions generated by passes"""

    def err(self, msg, n):
        """adds a compile exception to internal state"""
        self.errors.append(FrontendError(msg, n))


class Visitor:
    """visits each class, calling a custom visit function, if one is defined, for each
    node type"""

    def _find_custom_visit_func(self, node: Ast):
        for name, func in inspect.getmembers(type(self), inspect.isfunction):
            if not name.startswith("visit") or name == "visit_default":
                # not a visitor, or the default visit func
                continue
            signature = inspect.signature(func)
            params = list(signature.parameters.values())
            assert len(params) == 3
            assert params[1].annotation is not None
            annotations = typing.get_type_hints(func)
            param_type = annotations[params[1].name]
            if is_instance_compat(node, param_type):
                return getattr(self, name)
        # call the default
        return self.visit_default

    def _visit(self, node: Ast, state: FrontendState):
        visit_func = self._find_custom_visit_func(node)
        visit_func(node, state)

    def visit_default(self, node: Ast, state: FrontendState):
        pass

    def run(self, start: Ast, state: FrontendState):
        """runs the visitor, starting at the given node, descending depth-first"""

        def _descend(node: Ast):
            if not isinstance(node, Ast):
                return
            children = []
            for field in fields(node):
                field_val = getattr(node, field.name)
                if isinstance(field_val, list):
                    children.extend(field_val)
                else:
                    children.append(field_val)

            for child in children:
                if not isinstance(child, Ast):
                    continue
                _descend(child)
                if len(state.errors) != 0:
                    break
                self._visit(child, state)
                if len(state.errors) != 0:
                    break

        _descend(start)
        self._visit(start, state)


class Transformer(Visitor):

    class Delete:
        pass

    def run(self, start: Ast, state: CompileState):

        def _descend(node):
            if not isinstance(node, Ast):
                return
            for field in fields(node):
                field_val = getattr(node, field.name)
                if isinstance(field_val, list):
                    # child is a list, iterate over each member of the list
                    # use a copy so we can remove as we traverse, also so
                    # we don't visit things that we added
                    for idx, child in enumerate(field_val[:]):
                        if not isinstance(child, Ast):
                            continue
                        _descend(child)
                        if len(state.errors) != 0:
                            break
                        transformed = self._visit(child, state)
                        if len(state.errors) != 0:
                            break
                        if isinstance(transformed, Iterable):
                            assert all(
                                isinstance(n, Ast) for n in transformed
                            ), transformed
                            # func split one node into many
                            # remove the original child and add the new ones
                            # insert them in the place where the child used to be, in the right order
                            field_val.remove(child)
                            for new_child_idx, new_child in enumerate(transformed):
                                field_val.insert(idx + new_child_idx, new_child)
                        elif isinstance(transformed, Ast):
                            field_val.remove(child)
                            field_val.insert(idx, transformed)
                        elif transformed is Transformer.Delete:
                            # just delete it
                            field_val.remove(child)
                        else:
                            assert transformed is None, transformed
                            # don't do anything, didn't return anything
                        # update the field
                    setattr(node, field.name, field_val)
                else:
                    if isinstance(transformed, Ast):
                        setattr(node, field.name, transformed)
                    elif transformed is Transformer.Delete:
                        # just delete it
                        setattr(node, field.name, None)
                    else:
                        assert transformed is None, transformed
                        # don't do anything, didn't return anything

        _descend(start)
        self._visit(start, state)


class TopDownVisitor(Visitor):

    def run(self, start: Ast, state: FrontendState):
        """runs the visitor, starting at the given node, descending breadth-first"""

        def _descend(node: Ast):
            if not isinstance(node, Ast):
                return
            children = []
            for field in fields(node):
                field_val = getattr(node, field.name)
                if isinstance(field_val, list):
                    children.extend(field_val)
                else:
                    children.append(field_val)

            for child in children:
                if not isinstance(child, Ast):
                    continue
                self._visit(child, state)
                if len(state.errors) != 0:
                    break
                _descend(child)
                if len(state.errors) != 0:
                    break

        self._visit(start, state)
        _descend(start)
