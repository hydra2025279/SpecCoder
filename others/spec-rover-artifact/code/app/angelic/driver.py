"""
Instrument a specific method to output input/output values.

The values should be comprehensive, and include all necessary states.
"""

import os
from tempfile import NamedTemporaryFile

from loguru import logger

from app.angelic.data_structures import AngelicRequest
from app.task import SweTask


def instrument_batch_in_file(
    abs_file_path: str, requests_with_files: list[tuple[AngelicRequest, str]]
) -> None:
    # (1) sort the requests, based on function start line
    sorted_requests = sorted(requests_with_files, key=lambda x: x[0].start)

    # (2) read original lines
    with open(abs_file_path) as f:
        orig_lines = f.readlines()

    # (3) prepare preamble and figure out where to insert preamble
    code_preamble = []
    cur_dir = os.path.dirname(__file__)
    code_preamble.append("import sys\n")
    code_preamble.append(f"sys.path.append('{cur_dir}')\n")
    code_preamble.append("from acr_instrument_util import trace_expressions\n")

    idx_insert_preamble = 0
    for idx, line in enumerate(orig_lines):
        # we want to put our imports after all the __future__ imports
        if not line.startswith("import") and not line.startswith("from"):
            # this line may be comments or empty lines
            continue
        elif "__future__" in line:
            # skip `from __future__ import ...` lines, since those need to be
            # at the beginning of the file
            idx_insert_preamble = idx + 1
        else:
            # this means we have met some normal import lines
            # __future__ must appear before other imports, so it's safe to
            # break when we see a normal import
            break

    # (4) for each function, figure out what should be instrumented as decorator
    decorator_instruments = []
    for request, outfile_name in sorted_requests:
        # find out indentation level
        start_line_idx = request.start - 1
        start_line_content = orig_lines[start_line_idx]
        num_indent = len(start_line_content) - len(start_line_content.lstrip())
        indent = " " * num_indent

        # prepare code to instrument
        expr_list = request.expr_list
        code = f'{indent}@trace_expressions({expr_list}, "{outfile_name}")\n'

        # add this to the list
        decorator_instruments.append((start_line_idx, code))

    logger.info(
        f"<angelic driver> Composed instrumentation code: {decorator_instruments}"
    )

    # (5) ready to form the new file content and write it!
    new_lines = orig_lines[:idx_insert_preamble] + code_preamble
    chunk_start = idx_insert_preamble

    for idx, code in decorator_instruments:
        new_lines += orig_lines[chunk_start:idx] + [code]
        chunk_start = idx

    new_lines += orig_lines[chunk_start:]

    with open(abs_file_path, "w") as f:
        f.writelines(new_lines)

    ## DEBUG
    # base_name = os.path.basename(abs_file_path)
    # debug_file = os.path.join("/media/media0/specrover/acr-plus/", f"debug_{base_name}")
    # with open(debug_file, "w") as f:
    #     f.writelines(new_lines)


def instrument(requests_with_files: list[tuple[AngelicRequest, str]]) -> bool:
    """
    Instrument multiple AngelicRequests at the same time.
    When multiple AngelicRequests are for one file, the instrumentation is more complicated.

    Returns:
        - whether instrumentation is ok.
    """
    # print(f"Instrumenting the location: {bug_location}")

    logger.info(
        f"<agenlic driver> Final requests to instrument: {requests_with_files}."
    )

    if not requests_with_files:
        return False

    # group requests based on file_path
    grouped_requests: dict[str, list[tuple[AngelicRequest, str]]] = {}
    for request, outfile_name in requests_with_files:
        file_path = request.abs_file
        if file_path in grouped_requests:
            grouped_requests[file_path].append((request, outfile_name))
        else:
            grouped_requests[file_path] = [(request, outfile_name)]

    for file_path, requests_with_files in grouped_requests.items():
        instrument_batch_in_file(file_path, requests_with_files)
        logger.info(f"<angelic driver> Instrumented file {file_path} !")

    return True


def instrument_and_trace(
    task: SweTask,
    reproduced_test_content: str,
    angelic_requests: list[AngelicRequest],
) -> str | None:
    """
    Returns:
        - None if the tracing failed.
    """
    # (1) instrument with tracing/printing
    requests_with_files: list[tuple[AngelicRequest, str]] = []
    for request in angelic_requests:
        # create a file for each request
        instru_outfile = NamedTemporaryFile(
            mode="a", prefix="acr_instrumentation_output_", delete=False
        )
        instru_outfile.close()
        requests_with_files.append((request, instru_outfile.name))

    # print(f"\n\n\n\nInstrumenting with output file: {instru_outfile.name}")

    instrument_ok = instrument(requests_with_files)

    logger.info(f"<angelic driver> Code instrument result: {instrument_ok}")

    if not instrument_ok:
        task.reset_project()
        return None

    # (2) run reproducer on the instrumented program
    task.execute_reproducer(reproduced_test_content)

    instru_output = ""
    # read and concat content from all output files
    for _, instru_outfile_name in requests_with_files:
        if not os.path.exists(instru_outfile_name):
            # file not there -> nothing was written to it!
            continue

        with open(instru_outfile_name) as f:
            instru_output += f.read() + "\n"
        os.remove(instru_outfile_name)

    # instru_output can only contain meaningless characters
    instru_output = instru_output.strip()

    if not instru_output:
        instru_output = None

    logger.info(f"<angelic driver> Instrumentation output: {instru_output}")

    # at the end, restore the instrumentation
    task.reset_project()

    return instru_output
