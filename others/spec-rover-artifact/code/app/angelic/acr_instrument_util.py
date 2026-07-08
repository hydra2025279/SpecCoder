import inspect
import os
import sys
from types import FrameType
from typing import Any


class StateTracker:
    def __init__(self, func_name, file_name, expr_list, out_file):
        # name of the function to be tracked
        self.func_name = func_name
        self.file_name = file_name
        # self.last_vars: dict[str, Any] = {}
        self.expr_list = expr_list
        self.out_file = out_file
        if os.path.exists(out_file):
            os.remove(out_file)

    def log(self, content):
        # print(content)
        to_write = f"{content}\n"
        with open(self.out_file, "a") as f:
            f.write(to_write)
            f.flush()

    def file_name_fuzzy_equal(self, another_name):
        if self.file_name.endswith(another_name):
            return True
        if another_name.endswith(self.file_name):
            return True
        return False

    def traceit(self, frame: FrameType, event: str, arg: Any):
        if frame.f_code.co_name != self.func_name:
            # print(f"self.func is {self.func_name}")
            # print(f"Skipping {frame.f_code.co_name}")
            # print("Skipping due to func_name mismatch")
            return self.traceit

        # in large projects, we can have multiple functions with the same name
        # so also check if we are in the same file
        if not self.file_name_fuzzy_equal(frame.f_code.co_filename):
            # print(
            #     f"Skipping due to filename. Self: {self.file_name}, curr: {frame.f_code.co_filename}"
            # )
            return self.traceit

        if event == "call":
            self.log(f"Beginning the trace of {frame.f_code.co_name}()")

        if event == "line":
            try:
                module = inspect.getmodule(frame.f_code)
                if module is None:
                    source = inspect.getsource(frame.f_code)
                else:
                    source = inspect.getsource(module)
                current_line = source.split("\n")[frame.f_lineno - 1]

            except OSError as err:
                self.log(f"{err.__class__.__name__}: {err}")
                current_line = ""

            # NOTE: do not add line number in the execution trace, since we have
            # instrumented the source code and that changed the line numbers
            # log thread id
            # self.log(f"Thread ID: {threading.get_ident()}")
            self.log(current_line)

            # for line, print the values of the expressions
            local_vars = frame.f_locals
            expr_val_str = ""

            for expr in self.expr_list:
                try:
                    value = eval(expr, frame.f_globals, local_vars)
                    expr_val_str += f"{expr} = {value},    "
                except Exception:
                    # error happens when evaluating the expression at
                    # some location -> just skip
                    pass
            self.log(f"{' ' * 40} # {expr_val_str}")

        if event == "return":
            self.log(frame.f_code.co_name + "()" + " returns " + repr(arg))

        return self.traceit


def trace_expressions(expr_list, out_file):
    def decorator(func):
        def wrapper(*args, **kwargs):

            # print("==================> Inside decorator")

            file_name = inspect.getfile(func)
            state_tracker = StateTracker(func.__name__, file_name, expr_list, out_file)

            original_trace = sys.gettrace()
            sys.settrace(state_tracker.traceit)

            # Call the original function
            result = func(*args, **kwargs)

            sys.settrace(original_trace)

            return result

        return wrapper

    return decorator
