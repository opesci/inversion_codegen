from devito.function_manager import FunctionManager, FunctionDescriptor
from devito.compiler import get_tmp_dir, get_compiler_from_env
from devito.compiler import jit_compile_and_load, IntelMICCompiler
import cgen_wrapper as cgen
from codeprinter import ccode
import numpy as np
from sympy import symbols, IndexedBase, Indexed
from function_manager import FunctionDescriptor
from at_controller import get_best_best_block_size, get_optimal_block_size
from collections import Iterable
from os import path
from random import randint
from hashlib import sha1


class Propagator(object):
    """Propagator objects derive and encode C kernel code according
    to a given set of stencils, variables and time-stepping options.

    :param name: Name of the propagator kernel
    :param nt: Number of timesteps to execute
    :param shape: Shape of the data buffer over which to execute
    :param spc_border: Number of spatial padding layers
    :param time_order: Order of the time discretisation
    :param forward: Flag indicating whether to execute forward in time
    :param compiler: Compiler class used to perform JIT compilation.
                     If not provided, the compiler will be inferred from the
                     environment variable DEVITO_ARCH, or default to GNUCompiler.
    :param profile: Flag to enable performance profiling
    :param cache_blocking: Flag to enable cache blocking
    :param block_size: Block size used for cache clocking
    """
    def __init__(self, name, nt, shape, spc_border=0, time_order=0,
                 forward=True, compiler=None, profile=False,
                 cache_blocking=False, block_size=5, auto_tune=False):
        # Derive JIT compilation infrastructure
        self.compiler = compiler or get_compiler_from_env()
        self.mic_flag = isinstance(self.compiler, IntelMICCompiler)

        self.t = symbols("t")

        self.cache_blocking = cache_blocking
        self.auto_tune = auto_tune

        self.spc_order = spc_border * 2  # setting space order
        self.time_order = time_order

        if self.cache_blocking:
            if self.auto_tune:  # if auto tuning get block optimal block sizes and tune around that
                self.tune_b_size = get_optimal_block_size(shape, self.time_order, self.spc_order)

                self.tune_range = 8  # range of tuning around tune_b_size
                if self.tune_b_size - self.tune_range <= 0:  # making sure that tune_b - tune_range > 0
                    self.tune_range -= - (self.tune_b_size - self.tune_range) + 1

                block_size = self.tune_b_size
            else:  # else check if there is best block_size from at report else use optimal one
                optimal_block_size = get_optimal_block_size(shape, self.time_order, self.spc_order)
                block_size = get_best_best_block_size(self.time_order, self.spc_order)

                if block_size:
                    block_size.append(optimal_block_size)  # append outer most dimension as it is not auto tuned
                else:
                    block_size = optimal_block_size  # use optimal block size
        try:
        if(isinstance(block_size, Iterable)):
            if(len(block_size) == len(shape)):
                self.block_sizes = block_size
            else:
                raise ArgumentError("Block size should either be a single number or an array of the same size as the spatial domain")
        else:
            # A single block size has been passed. Broadcast it to a list of the size of shape
            self.block_sizes = [int(block_size)]*len(shape)
        # We assume the dimensions are passed to us in the following order
        # var_order
        if len(shape) == 2:
            self.space_dims = symbols("x z")
        else:
            self.space_dims = symbols("x y z")
        self.space_dims = self.space_dims[0:len(shape)]
        self.loop_counters = symbols("i1 i2 i3 i4")
        self._pre_kernel_steps = []
        self._post_kernel_steps = []
        self._forward = forward
        self.prep_variable_map()
        self.t_replace = {}
        self.time_steppers = []
        self.nt = nt
        self.time_loop_stencils_b = []
        self.time_loop_stencils_a = []
        # Start with the assumption that the propagator needs to save the field in memory at every time step
        self._save = True
        # This might be changed later when parameters are being set
        self.profile = profile
        # Which function parameters need special (non-save) time stepping and which don't
        self.save_vars = {}
        self.fd = FunctionDescriptor(name)
        self.pre_loop = []
        self.post_loop = []

        if self.profile:
            self.add_local_var("time", "double")
            self.pre_loop.append(cgen.Statement("struct timeval start, end"))
            self.pre_loop.append(cgen.Assign("time", 0))
            self.post_loop.append(cgen.PrintStatement("time: %f\n", "time"))

        # Auto tuning
        if self.cache_blocking and self.auto_tune:
            self.at_markers = [("M1_start", "M1_end"), ("M2_start", "M2_end")]  # markers for at pragmas
            self._at_init_block_vars()

        if forward:
            self._time_step = 1
        else:
            self._time_step = -1
        self._space_loop_limits = {}
        for i, dim in enumerate(self.space_dims):
                self._space_loop_limits[dim] = (spc_border, shape[i] - spc_border)

        # Kernel operation intensity dictionary
        self._kernel_dic_oi = {'add': 0, 'mul': 0, 'load': 0, 'store': 0, 'load_list': [], 'load_all_list': [], 'oi_high': 0, 'oi_high_weighted': 0, 'oi_low': 0, 'oi_low_weighted': 0}

        # Cache C code, lib and function objects
        self._ccode = None
        self._lib = None
        self._cfunction = None

    @property
    def basename(self):
        """Generate a unique basename path for auto-generated files.

        The basename is generated by hashing grid variables (fd.params)
        and made unique by the addition of a random salt value.
        """
        string = "%s-%s" % (str(self.fd.params), randint(0, 100000000))
        return path.join(get_tmp_dir(), sha1(string).hexdigest())

    @property
    def ccode(self):
        """Returns the auto-generated C code as a string"""
        if self._ccode is None:
            manager = FunctionManager([self.fd], mic_flag=self.mic_flag,
                                      openmp=self.compiler.openmp)
            # For some reason we need this call to trigger fd.body
            self.get_fd()
            self._ccode = str(manager.generate())
        return self._ccode

    @property
    def cfunction(self):
        """Returns the JIT-compiled C function as a ctypes.FuncPtr object

        Note that this invokes the JIT compilation toolchain with the
        compiler class derived in the constructor.
        """
        if self._lib is None:
            self._lib = jit_compile_and_load(self.ccode, self.basename,
                                             self.compiler)
        if self._cfunction is None:
            self._cfunction = getattr(self._lib, self.fd.name)
            if not self.mic_flag:
                self._cfunction.argtypes = self.fd.argtypes
        return self._cfunction

    @property
    def save(self):
        return self._save

    @save.setter
    def save(self, save):
        if save is not True:
            self.time_steppers = [symbols("t%d" % i) for i in range(self.time_order+1)]
            self.t_replace = {}
            for i, t_var in enumerate(reversed(self.time_steppers)):
                self.t_replace[self.t - i*self._time_step] = t_var
        self._save = self._save and save

    @property
    def time_loop_limits(self):
        if self._forward:
            loop_limits = (0, self.nt)
        else:
            loop_limits = (self.nt-1, -1)
        return loop_limits

    def prep_variable_map(self):
        """ Mapping from model variables (x, y, z, t) to loop variables (i1, i2, i3, i4)
        For now, i1 i2 i3 are assigned in the order the variables were defined in init( #var_order)
        """
        var_map = {}
        i = 0
        for dim in self.space_dims:
            var_map[dim] = symbols("i%d" % (i + 1))
            i += 1
        var_map[self.t] = symbols("i%d" % (i + 1))
        self._var_map = var_map

    def sympy_to_cgen(self, subs, stencils, stencil_args):
        stmts = []
        for equality, args in zip(stencils, stencil_args):
            equality = equality.subs(dict(zip(subs, args)))
            self._kernel_dic_oi = self._get_ops_expr(equality.rhs, self._kernel_dic_oi, False)
            self._kernel_dic_oi = self._get_ops_expr(equality.lhs, self._kernel_dic_oi, True)
            stencil = self.convert_equality_to_cgen(equality)
            stmts.append(stencil)
        kernel = self._pre_kernel_steps
        kernel += stmts
        kernel += self._post_kernel_steps
        return cgen.Block(kernel)

    def convert_equality_to_cgen(self, equality):
        if isinstance(equality, cgen.Generable):
            return equality
        else:
            return cgen.Assign(ccode(self.time_substitutions(equality.lhs).xreplace(self._var_map)), ccode(self.time_substitutions(equality.rhs).xreplace(self._var_map)))

    def generate_loops(self, loop_body):
        """ Assuming that the variable order defined in init (#var_order) is the
        order the corresponding dimensions are layout in memory, the last variable
        in that definition should be the fastest varying dimension in the arrays.
        Therefore reverse the list of dimensions, making the last variable in
        #var_order (z in the 3D case) vary in the inner-most loop
        """
        # Space loops
        if self.cache_blocking:
            loop_body = self.generate_space_loops_blocking(loop_body)
        else:
            loop_body = self.generate_space_loops(loop_body)

        omp_master = [cgen.Pragma("omp master")] if self.compiler.openmp else []
        omp_single = [cgen.Pragma("omp single")] if self.compiler.openmp else []
        omp_parallel = [cgen.Pragma("omp parallel")] if self.compiler.openmp else []
        omp_for = [cgen.Pragma("omp for")] if self.compiler.openmp else []
        t_loop_limits = self.time_loop_limits
        t_var = str(self._var_map[self.t])
        cond_op = "<" if self._forward else ">"

        if self.save is not True:
            # To cycle between array elements when we are not saving time history
            time_stepping = self.get_time_stepping()
        else:
            time_stepping = []
        loop_body = omp_for + [loop_body] if self.compiler.openmp else [loop_body]
        # Statements to be inserted into the time loop before the spatial loop
        time_loop_stencils_b = [self.convert_equality_to_cgen(x) for x in self.time_loop_stencils_b]
        # Statements to be inserted into the time loop after the spatial loop
        time_loop_stencils_a = [self.convert_equality_to_cgen(x) for x in self.time_loop_stencils_a]
        if self.profile:
            start_time_stmt = omp_master + [cgen.Block([cgen.Statement("gettimeofday(&start, NULL)")])]
            end_time_stmt = omp_master + [cgen.Block([cgen.Statement("gettimeofday(&end, NULL)")] + [cgen.Statement("time += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6")])]
        else:
            start_time_stmt = []
            end_time_stmt = []
        initial_block = omp_single + [cgen.Block(time_stepping + time_loop_stencils_b)] if time_stepping or time_loop_stencils_b else []
        initial_block = initial_block + start_time_stmt
        end_block = end_time_stmt + omp_single + [cgen.Block(time_loop_stencils_a)] if time_loop_stencils_a else end_time_stmt
        loop_body = cgen.Block(initial_block + loop_body + end_block)
        loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", t_var), str(t_loop_limits[0])), t_var + cond_op + str(t_loop_limits[1]), t_var + "+=" + str(self._time_step), loop_body)
        # Code to declare the time stepping variables (outside the time loop)
        def_time_step = [cgen.Value("int", t_var_def.name) for t_var_def in self.time_steppers]
        body = def_time_step + self.pre_loop + omp_parallel + [loop_body] + self.post_loop
        return cgen.Block(body)

    def generate_space_loops(self, loop_body):
        ivdep = True
        for spc_var in reversed(list(self.space_dims)):
            dim_var = self._var_map[spc_var]
            loop_limits = self._space_loop_limits[spc_var]
            loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", dim_var),
                                                        str(loop_limits[0])),
                                 str(dim_var) + "<" + str(loop_limits[1]), str(dim_var) + "++", loop_body)
            if ivdep and len(self.space_dims) > 1:
                loop_body = cgen.Block([self.compiler.pragma_ivdep] + [loop_body])
            ivdep = False
        return loop_body

    def generate_space_loops_blocking(self, loop_body):
        ivdep = True
        remainder = False
        orig_loop_body = loop_body

        inner_most_dim_passed = False  # used in oder to avoid at inner most dimension

        for spc_var, block_size in reversed(zip(list(self.space_dims), self.block_sizes)):
            dim_var = str(self._var_map[spc_var])

            if self.auto_tune and inner_most_dim_passed:  # change block size into var
                block_size = dim_var + "block"
            else:
                inner_most_dim_passed = True

            block_var = dim_var + "b"
            loop_limits = self._space_loop_limits[spc_var]
            loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", dim_var),
                                                        block_var),
                                 dim_var + "<" + block_var+"+"+str(block_size), dim_var + "++", loop_body)
            if ivdep and len(self.space_dims) > 1:
                loop_body = cgen.Block([self.compiler.pragma_ivdep] + [loop_body])
            ivdep = False

        inner_most_dim_passed = False  # used in oder to avoid at inner most dimension

        for spc_var, block_size in reversed(zip(list(self.space_dims), self.block_sizes)):
            orig_var = str(self._var_map[spc_var])
            dim_var = orig_var + "b"
            loop_limits = self._space_loop_limits[spc_var]
            old_upper_limit = loop_limits[1]
            new_upper_limit = old_upper_limit-old_upper_limit % block_size
            if old_upper_limit - new_upper_limit > 0:
                remainder = True
            loop_limits = (loop_limits[0], new_upper_limit)

            if self.auto_tune and inner_most_dim_passed:  # change block size into var name
                block_size = orig_var + "block"
            else:
                inner_most_dim_passed = True

            loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", dim_var),
                                                        str(loop_limits[0])),
                                 str(dim_var) + "<" + str(loop_limits[1]), str(dim_var) + "+=" + str(block_size), loop_body)
        if remainder:
            remainder_loop = orig_loop_body
            for spc_var, block_size in reversed(zip(list(self.space_dims), self.block_sizes)):
                dim_var = str(self._var_map[spc_var])
                loop_limits = self._space_loop_limits[spc_var]
                old_upper_limit = loop_limits[1]
                new_upper_limit = old_upper_limit-old_upper_limit % block_size
                loop_limits = (new_upper_limit, old_upper_limit)
                remainder_loop = cgen.For(cgen.InlineInitializer(cgen.Value("int", dim_var), str(loop_limits[0])),
                                          str(dim_var) + "<" + str(loop_limits[1]), str(dim_var) + "++", remainder_loop)
                if ivdep and len(self.space_dims) > 1:
                    loop_body = cgen.Block([self.compiler.pragma_ivdep] + [loop_body])
                ivdep = False
            loop_body = cgen.Block([loop_body, remainder_loop])

        return loop_body

    def add_loop_step(self, assign, before=False):
        stm = self.convert_equality_to_cgen(assign)
        if before:
            self._pre_kernel_steps.append(stm)
        else:
            self._post_kernel_steps.append(stm)

    def add_devito_param(self, param):
        save = True
        if hasattr(param, "save"):
            save = param.save
        self.add_param(param.name, param.shape, param.dtype, save)

    def add_param(self, name, shape, dtype, save=True):
        self.fd.add_matrix_param(name, shape, dtype)
        self.save = save
        self.save_vars[name] = save
        return IndexedBase(name, shape=shape)

    def add_scalar_param(self, name, dtype):
        self.fd.add_value_param(name, dtype)
        return symbols(name)

    def add_local_var(self, name, dtype):
        self.fd.add_local_variable(name, dtype)
        return symbols(name)

    def get_fd(self):
        """Get a FunctionDescriptor that describes the code represented by this Propagator
        in the format that FunctionManager and JitManager can deal with it. Before calling,
        make sure you have either called set_jit_params or set_jit_simple already.
        """
        try:  # Assume we have been given a a loop body in cgen types
            self.fd.set_body(self.generate_loops(self.loop_body))
        except:  # We might have been given Sympy expression to evaluate
            # This is the more common use case so this will show up in error messages
            self.fd.set_body(self.generate_loops(self.sympy_to_cgen(self.subs, self.stencils, self.stencil_args)))
        return self.fd

    def get_time_stepping(self):
        ti = self._var_map[self.t]
        body = []
        time_stepper_indices = range(self.time_order+1)
        first_time_index = 0
        step_backwards = -1
        if self._forward is not True:
            time_stepper_indices = reversed(time_stepper_indices)
            first_time_index = self.time_order
            step_backwards = 1
        for i in time_stepper_indices:
            lhs = self.time_steppers[i].name
            if i == first_time_index:
                rhs = ccode(ti % (self.time_order+1))
            else:
                rhs = ccode((self.time_steppers[i+step_backwards]+1) % (self.time_order+1))
            body.append(cgen.Assign(lhs, rhs))

        return body

    def time_substitutions(self, sympy_expr):
        """This method checks through the sympy_expr to replace the time index with a cyclic index
        but only for variables which are not being saved in the time domain
        """
        if isinstance(sympy_expr, Indexed):
            array_term = sympy_expr
            if not str(array_term.base.label) in self.save_vars:
                raise(ValueError, "Invalid variable '%s' in sympy expression. Did you add it to the operator's params?" % str(array_term.base.label))
            if not self.save_vars[str(array_term.base.label)]:
                array_term = array_term.xreplace(self.t_replace)
            return array_term
        else:
            for arg in sympy_expr.args:
                sympy_expr = sympy_expr.subs(arg, self.time_substitutions(arg))
        return sympy_expr

    def add_time_loop_stencil(self, stencil, before=False):
        if before:
            self.time_loop_stencils_b.append(stencil)
        else:
            self.time_loop_stencils_a.append(stencil)

            # initialises block sizes as variables and adds auto tuning pragmas

    def _at_init_block_vars(self):
        block_vars = []  # Blocking var names

        for i in range(0, len(self.space_dims) - 1):  # generate block size vars. We want to ignore inner most dimension
            block_vars.append(str(self.loop_counters[i]) + "block")

        # main auto tuning pragma
        at_main_pragma = "isat tuning name(spc_o_%s_tm_o_%s) scope(%s, %s) measure(%s, %s)" \
                         % (self.spc_order, self.time_order, self.at_markers[0][0], self.at_markers[0][1],
                            self.at_markers[1][0], self.at_markers[1][1])

        for block_var in block_vars:  # appends vars that we want to tune to main at pragma
            at_main_pragma += " variable(%s, range(%d, %d, 1))" \
                              % (block_var, self.tune_b_size - self.tune_range, self.tune_b_size + self.tune_range)

        # dependant will try all possible permutations of block sizes // prob what we want
        # independant will find the first optimal varthen progress searching for the next one using the one found before
        at_main_pragma += " search(dependent)"

        self.pre_loop.append(cgen.Pragma(at_main_pragma))

        for block_var in block_vars:
            # init the block size variables
            self.pre_loop.append(cgen.Statement("int const %s = %d" % (block_var, self.tune_b_size)))

        # Setting auto tuning scope
        self.pre_loop.append(cgen.Pragma("isat marker %s" % self.at_markers[0][0]))
        self.post_loop.append(cgen.Pragma("isat marker %s" % self.at_markers[0][1]))

    def _get_ops_expr(self, expr, dict1, is_lhs=False):
        """Get number of different operations in expression expr.
        Types of operations are ADD (inc -) and MUL (inc /), arrays (IndexedBase objects) in expr that are not in list arrays
        are added to the list. Return dictionary of (#ADD, #MUL, list of unique names of fields, list of unique field elements)
        """
        result = dict1  # dictionary to return
        # add array to list arrays if it is not in it
        if isinstance(expr, Indexed):
                base = expr.base.label
                if is_lhs:
                        result['store'] += 1
                if base not in result['load_list']:
                        result['load_list'] += [base]  # accumulate distinct array
                if expr not in result['load_all_list']:
                        result['load_all_list'] += [expr]  # accumulate distinct array elements
                return result

        if expr.is_Mul or expr.is_Add or expr.is_Pow:
                args = expr.args
                # increment MUL or ADD by # arguments less 1
                # sympy multiplication and addition can have multiple arguments
                if expr.is_Mul:
                        result['mul'] += len(args)-1
                else:
                        if expr.is_Add:
                                result['add'] += len(args)-1
                # recursive call of all arguments
                for expr2 in args:
                        result2 = self._get_ops_expr(expr2, result, is_lhs)

                return result2
        # return zero and unchanged array if execution gets here
        return result

    def get_kernel_oi(self, dtype=np.float32):
        """Get the operation intensity of the kernel. The types of operations are ADD (inc -), MUL (inc /), LOAD, STORE.
        #LOAD = number of unique fields in the kernel. The function returns tuple (#ADD, #MUL, #LOAD, #STORE)
        Wold_size is given by dtype
        Operation intensity OI = (ADD+MUL)/[(LOAD+STORE)*word_size]
        Weighted OI, OI_w = (ADD+MUL)/(2*Max(ADD,MUL)) * OI
        """
        load = 0
        load_all = 0
        word_size = np.dtype(dtype).itemsize
        load += len(self._kernel_dic_oi['load_list'])
        store = self._kernel_dic_oi['store']
        load_all += len(self._kernel_dic_oi['load_all_list'])
        self._kernel_dic_oi['load'] = load_all
        add = self._kernel_dic_oi['add']
        mul = self._kernel_dic_oi['mul']
        self._kernel_dic_oi['oi_high'] = float(add+mul)/(load+store)/word_size
        self._kernel_dic_oi['oi_high_weighted'] = self._kernel_dic_oi['oi_high']*(add+mul)/max(add, mul)/2.0
        self._kernel_dic_oi['oi_low'] = float(add+mul)/(load_all+store)/word_size
        self._kernel_dic_oi['oi_low_weighted'] = self._kernel_dic_oi['oi_low']*(add+mul)/max(add, mul)/2.0

        return self._kernel_dic_oi['oi_high']
