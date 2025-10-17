from fprime_gds.common.fpy.bytecode.directives import (
    Directive,
    GotoDirective,
    IfDirective,
)
from fprime_gds.common.fpy.error import BackendError, FrontendError
from fprime_gds.common.fpy.backend_types import (
    BackendState,
    IrBasicBlock,
    IrDirective,
    IrFunction,
    IrGoto,
    IrIf,
    IrInstruction,
    IrModule,
    IrStmt,
    IrVisitor,
)


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
        self,
        stmt: IrInstruction,
        block: IrBasicBlock,
        func: IrFunction,
        state: BackendState,
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
