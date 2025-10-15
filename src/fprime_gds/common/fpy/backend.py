from fprime_gds.common.fpy.backend_types import BackendState
from fprime_gds.common.fpy.bytecode.directives import (
    Directive,
    GotoDirective,
    IfDirective,
)
from fprime_gds.common.fpy.error import BackendError, FrontendError
from fprime_gds.common.fpy.ir import (
    IrBasicBlock,
    IrBinaryOp,
    IrDirective,
    IrInst,
    IrFunction,
    IrGoto,
    IrGotoLabel,
    IrIf,
    IrInstruction,
    IrModule,
    IrStmt,
    IrVisitor,
)


def get_64_bit_numeric_type(type: FppType) -> FppType:
    assert type in SPECIFIC_NUMERIC_TYPES, type
    return (
        I64Type
        if type in SIGNED_INTEGER_TYPES
        else U64Type if type in UNSIGNED_INTEGER_TYPES else F64Type
    )


def convert_numeric_type(from_type: FppType, to_type: FppType) -> list[Directive]:
    if from_type == to_type:
        return []

    # only valid runtime type conversion is between two numeric types
    assert from_type in SPECIFIC_NUMERIC_TYPES and to_type in SPECIFIC_NUMERIC_TYPES, (
        from_type,
        to_type,
    )
    # also invalid to convert from a float to an integer at runtime due to loss of precision
    assert not (
        from_type in SPECIFIC_FLOAT_TYPES and to_type in SPECIFIC_INTEGER_TYPES
    ), (
        from_type,
        to_type,
    )

    dirs = []
    # first go to 64 bit width
    dirs.extend(extend_numeric_type_to_64_bits(from_type))
    from_64_bit = get_64_bit_numeric_type(from_type)
    to_64_bit = get_64_bit_numeric_type(to_type)

    # now convert from int to float if necessary
    if from_64_bit == U64Type and to_64_bit == F64Type:
        dirs.append(UnsignedIntToFloatDirective())
        from_64_bit = F64Type
    elif from_64_bit == I64Type and to_64_bit == F64Type:
        dirs.append(SignedIntToFloatDirective())
        from_64_bit = F64Type
    elif from_64_bit == U64Type or from_64_bit == I64Type:
        assert to_64_bit == U64Type or to_64_bit == I64Type
        # conversion from signed to unsigned int is implicit, doesn't need code gen
        from_64_bit = to_64_bit

    assert from_64_bit == to_64_bit, (from_64_bit, to_64_bit)

    # now truncate back down to desired size
    dirs.extend(truncate_numeric_type_from_64_bits(to_64_bit, to_type.getMaxSize()))
    return dirs


def truncate_numeric_type_from_64_bits(
    from_type: FppType, new_size: int
) -> list[Directive]:

    assert new_size in (1, 2, 4, 8), new_size
    assert from_type.getMaxSize() == 8, from_type.getMaxSize()

    if new_size == 8:
        # already correct size
        return []

    if from_type == F64Type:
        # only one option for float trunc
        assert new_size == 4, new_size
        return [FloatTruncateDirective()]

    # must be an int
    assert issubclass(from_type, IntegerType), from_type

    if new_size == 1:
        return [IntegerTruncate64To8Directive()]
    elif new_size == 2:
        return [IntegerTruncate64To16Directive()]

    return [IntegerTruncate64To32Directive()]


def extend_numeric_type_to_64_bits(type: FppType) -> list[Directive]:
    if type.getMaxSize() == 8:
        # already 8 bytes
        return []
    if type == F32Type:
        return [FloatExtendDirective()]

    # must be an int
    assert issubclass(type, IntegerType), type

    from_size = type.getMaxSize()
    assert from_size in (1, 2, 4, 8), from_size

    if type in SIGNED_INTEGER_TYPES:
        if from_size == 1:
            return [IntegerSignedExtend8To64Directive()]
        elif from_size == 2:
            return [IntegerSignedExtend16To64Directive()]
        else:
            return [IntegerSignedExtend32To64Directive()]
    else:
        if from_size == 1:
            return [IntegerZeroExtend8To64Directive()]
        elif from_size == 2:
            return [IntegerZeroExtend16To64Directive()]
        else:
            return [IntegerZeroExtend32To64Directive()]


class AssignIds(IrVisitor):
    def __init__(self):
        self.next_id = 0

    def visit_stmt_default(self, inst, block, func, mod, state):
        inst.id = self.next_id
        self.next_id += 1


class CalculateLineNumbers(IrVisitor):
    def __init__(self):
        self.next_line_idx = 0

    def visit_IrInstruction(
        self, stmt: IrInstruction, block: IrBasicBlock, func: IrFunction, state: BackendState
    ):
        # each instr is guaranteed to map to exactly one dir
        state.dir_line_indices[stmt] = self.next_line_idx
        self.next_line_idx += 1

    def visit_IrGotoLabel(
        self,
        stmt: IrGotoLabel,
        block: IrBasicBlock,
        func: IrFunction,
        state: BackendState,
    ):
        state.func_goto_labels[func][stmt.label] = self.next_line_idx


class ResolveGotos(IrVisitor):

    def visit_IrGoto(
        self, inst: IrGoto, block: IrBasicBlock, func: IrFunction, state: BackendState
    ):
        line_idx = state.func_goto_labels[func].get(inst.label)
        if line_idx is None:
            state.err(f"Unknown label {inst.label}")
            return
        state.goto_indices[inst] = line_idx

    def visit_IrIf(
        self, inst: IrIf, block: IrBasicBlock, func: IrFunction, state: BackendState
    ):
        line_idx = state.func_goto_labels[func].get(inst.goto_false_label)
        if line_idx is None:
            state.err(f"Unknown label {inst.goto_false_label}")
            return
        state.goto_indices[inst] = line_idx


class GenerateBasicBlocks(IrVisitor):
    def visit_basic_block(self, block, func, mod, state):
        dirs = []
        for stmt in block.stmts:
            if isinstance(stmt, IrDirective):
                dirs.append(stmt.dir)
            elif isinstance(stmt, IrGoto):
                dir = GotoDirective(state.goto_indices[stmt])
                dirs.append(dir)
            elif isinstance(stmt, IrIf):
                dir = IfDirective(state.goto_indices[stmt])
                dirs.append(dir)
            else:
                assert not isinstance(stmt, IrInstruction), stmt

        state.block_dirs[block] = dirs


class GenerateFunctions(IrVisitor):
    def visit_function(self, func, mod, state):
        dirs = []
        for block in func.blocks:
            dirs.extend(state.block_dirs[block])
        state.func_dirs[func] = dirs


class GenerateModules(IrVisitor):
    def visit_module(self, mod, state):
        dirs = []
        for func in mod.funcs:
            dirs.extend(state.func_dirs[func])
        state.mod_dirs[mod] = dirs


def ir_to_bytecode(module: IrModule) -> list[Directive] | BackendError:
    passes: list[IrVisitor] = [
        AssignIds(),
        CalculateLineNumbers(),
        ResolveGotos(),
        GenerateBasicBlocks(),
        GenerateFunctions(),
        GenerateModules(),
    ]
    state = BackendState()
    for backend_pass in passes:
        backend_pass.run(module, state)
        if len(state.errors) != 0:
            return state.errors[0]

    if len(dirs) > MAX_DIRECTIVES_COUNT:
        err = FrontendError(
            f"Too many directives in sequence (expected less than {MAX_DIRECTIVES_COUNT}, had {len(dirs)})"
        )
        return err
    # TODO check lvar array not > max stack size (AND TEST THIS!)
    return state.mod_dirs[module]
