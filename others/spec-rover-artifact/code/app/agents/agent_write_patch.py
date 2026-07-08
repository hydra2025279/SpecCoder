"""
An agent, which is only responsible for the write_patch tool call.
"""

import json
from collections import defaultdict
from collections.abc import Generator
from copy import deepcopy
from os.path import join as pjoin
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeAlias

from loguru import logger

from app import config
from app.agents import agent_common
from app.agents.agent_angelic import AngelicAgent
from app.agents.agent_common import InvalidLLMResponse
from app.angelic.data_structures import AngelicMode, AngelicRequest, AngelicTransaction
from app.data_structures import BugLocation, MessageThread, SearchResult
from app.log import print_acr, print_patch_generation
from app.model import common
from app.post_process import (
    ExtractStatus,
    convert_response_to_diff,
    extract_diff_one_instance,
    record_extract_status,
)
from app.search.search_manage import SearchManager
from app.task import Task

SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
Another developer has already collected code context related to the issue for you.
Your task is to write a patch that resolves this issue.
Do not make changes to test files or write tests; you are only interested in crafting a patch.
REMEMBER:
- You should only make minimal changes to the code to resolve the issue.
- Your patch should preserve the program functionality as much as possible.
- In your patch, DO NOT include the line numbers at the beginning of each line!
"""


USER_PROMPT_INIT = """Write a patch for the issue, based on the relevant code context.
First explain the reasoning, and then write the actual patch.
When writing the patch, remember the following:
 - You do not have to modify every location provided - just make the necessary changes.
 - Pay attention to the addtional context as well - sometimes it might be better to fix there.
 - You should import necessary libraries if needed.

Return the patch in the format below.
Within `<file></file>`, replace `...` with actual file path.
Within `<original></original>`, replace `...` with the original code snippet from the program.
Within `<patched></patched>`, replace `...` with the fixed version of the original code.
When adding orignal code and updated code, pay attention to indentation, as the code is in Python.
You can write multiple modifications if needed.

Example format:

# modification 1
```
<file>...</file>
<original>...</original>
<patched>...</patched>
```

# modification 2
```
<file>...</file>
<original>...</original>
<patched>...</patched>
```

# modification 3
...
```
NOTE:
- In your patch, DO NOT include the line numbers at the beginning of each line!
- Inside <original> and </original>, you should provide the original code snippet from the program.
This original code snippet MUST match exactly to a continuous block of code in the original program,
since the system will use this to locate the code to be modified.
"""


PatchHandle: TypeAlias = str


class PatchAgent:
    EMPTY_PATCH_HANDLE = "EMPTY"

    def __init__(
        self,
        task: Task,
        search_manager: SearchManager,
        issue_stmt: str,
        context_thread: MessageThread,
        bug_locs: list[BugLocation],
        task_dir: str,
    ) -> None:
        self.task = task
        self.search_manager = search_manager
        self.issue_stmt = issue_stmt
        self.context_thread = context_thread  # the search conv historh thread
        # TODO: merge class_context_code into bug_loc_info, and make them one type
        self.bug_locs: list[BugLocation] = bug_locs
        self.task_dir = task_dir

        self._request_idx: int = -1
        self._responses: dict[PatchHandle, str] = {}
        self._diffs: dict[PatchHandle, str] = {}
        self._feedbacks: dict[PatchHandle, list[str]] = defaultdict(list)
        self._history: list[PatchHandle] = []

        self._angelic_transactions: dict[PatchHandle, list[AngelicTransaction]] = {}

    def write_applicable_patch_without_feedback(
        self, test_content: str | None = None, retries: int = 3
    ) -> tuple[PatchHandle, str]:
        return self._write_applicable_patch(
            max_feedbacks=0, retries=retries, test_content=test_content
        )

    def write_applicable_patch_with_feedback(
        self, test_content: str | None = None, max_feedbacks: int = 1, retries: int = 3
    ) -> tuple[PatchHandle, str]:
        return self._write_applicable_patch(
            max_feedbacks=max_feedbacks, retries=retries, test_content=test_content
        )

    def add_feedback(self, handle: PatchHandle, feedback: str) -> None:
        if handle not in self._diffs:
            raise ValueError("patch {} does not exist", handle)

        self._feedbacks[handle].append(feedback)

    def _write_applicable_patch(
        self, max_feedbacks: int, retries: int, test_content: str | None = None
    ) -> tuple[PatchHandle, str]:

        max_feedbacks = max_feedbacks if max_feedbacks >= 0 else len(self._history)
        num_feedbacks = min(max_feedbacks, len(self._history))
        history_handles = self._history[-num_feedbacks:]

        for _ in range(retries):
            applicable, response, diff_content, thread, angelic_tx = self._write_patch(
                history_handles, test_content=test_content
            )
            self._request_idx += 1
            print_patch_generation(response)
            Path(self.task_dir, f"patch_raw_{self._request_idx}.md").write_text(
                response
            )
            thread.save_to_file(
                Path(self.task_dir, f"conv_patch_{self._request_idx}.json")
            )

            if angelic_tx is not None:
                angelic_tx.save_to_file(
                    Path(
                        self.task_dir,
                        f"angelic_tx_{angelic_tx.mode.value}_{self._request_idx}.txt",
                    )
                )
                angelic_tx.save_json_to_file(
                    Path(
                        self.task_dir,
                        f"angelic_tx_{angelic_tx.mode.value}_{self._request_idx}.json",
                    )
                )

            msg = "Patch is applicable" if applicable else "Patch is not applicable"
            print_acr(msg)
            if applicable:
                print_acr(f"```diff\n{diff_content}\n```", "Extracted patch")

                angelic_txs = [angelic_tx] if angelic_tx else []
                handle = self._register_applicable_patch(
                    response, diff_content, angelic_txs
                )

                return handle, diff_content

        raise InvalidLLMResponse(
            f"Failed to write an applicable patch in {retries} attempts"
        )

    def _construct_angelic_select_prompt(self, angelic_mode: AngelicMode) -> str:
        prompt = (
            "To help you write patches, another developer has offered to perform some value-based analysis on the bug locations. "
            "If you think you need this additional information, you can request it. "
            "You can specify some locations and a list of expressions for each location that you are interested in. "
            "The developer will provide you with some analysis on how the values of the interesting expressions change "
            "in the buggy execution, as well as the expected values in the correct execution. "
            "(The correct execution is also called 'angelic execution.) "
            "This analysis should help you understand the bug and write a patch. "
            "\n\nNOTE:\n"
            "- You DO NOT need to investigate all the locations. Just pick "
            "those you think are relevant to the bug.\n"
            "- The expressions should be valid Python expressions in the corresponding method. "
            "If the expressions are not valid, they will be skipped.\n"
            "- For each location, specify ALL expressions that you think are relevant to the bug.\n"
            "\nYour response MUST follow the JSON format:"
            "\n```"
            "\n{"
            '\n    "investigate": ['
            "\n        {"
            '\n            "location": {,'
            '\n                "file": "...",'
            '\n                "class": "...",'
            '\n                "method": "..."'
            "\n            },"
            '\n            "expressions": ["expr_1", "expr_2", ...]'
            "\n        },"
            "\n        ..."
            "\n    ]"
            "\n}"
            "\n```"
            "\n\nwhere `location` is the location of the bug, and `expressions` is the list of expressions that you want to track. "
            "\n Inside `location`, you should provide the file path, class name, and method name, which should be from the bug locations provided earlier."
            "\n You can omit the `class` field if the method you want to investigate is outside a class.\n"
            "IMPORTANT:\n"
            "- For each location to investigate, you must provide a method name.\n"
            "- If you pick a location without a method name (i.e. the location is the entire class/file), that location will be skipped.\n"
            "- You can investigate multiple locations in a single response.\n"
        )

        # in REAL mode, agenlic analysis is always performed
        # However, in IMAGINARY mode, we let the agent decide whether to perform angelic analysis
        if angelic_mode == AngelicMode.IMAGINARY:
            prompt += (
                "- In general, a value-based analysis is helpful to understand the bug better. "
                "However, if you already have a clear understanding of the issue, "
                "and are confident in writing a correct patch to resolve the issue, "
                'you can skip this step by responding with `{"investigate": []}`.'
            )

        return prompt

    def _process_raw_angelic_request(
        self, angelic_to_investigate_resp: str
    ) -> list[AngelicRequest]:

        data = json.loads(angelic_to_investigate_resp)
        raw_requests = data["investigate"]

        all_angelic_requests: list[AngelicRequest] = []
        for raw_request in raw_requests:
            file = raw_request["location"].get("file", "")
            class_name = raw_request["location"].get("class", "")
            method_name = raw_request["location"].get("method", "")
            expr_list = raw_request["expressions"]

            if not method_name:  # skip
                continue

            if not file:  # skip
                continue

            if class_name:
                search_results: list[SearchResult]
                _, search_results, _ = (
                    self.search_manager.backend.search_method_in_class(
                        method_name, class_name
                    )
                )
                if not search_results:
                    continue
                for result in search_results:
                    if not result.file_path.endswith(file):
                        # file does not match
                        continue
                    if result.start is None or result.end is None:
                        # should not happen
                        continue
                    # matched file as well
                    request = AngelicRequest(
                        file,
                        result.file_path,
                        class_name,
                        method_name,
                        result.start,
                        result.end,
                        result.code,
                        expr_list,
                    )
                    all_angelic_requests.append(request)

            else:
                # no class name in request
                search_results: list[SearchResult]
                _, search_results, _ = (
                    self.search_manager.backend.search_method_in_file(method_name, file)
                )
                if not search_results:
                    continue

                for result in search_results:
                    if result.start is None or result.end is None:
                        # should not happen
                        continue
                    request = AngelicRequest(
                        file,
                        result.file_path,
                        None,
                        method_name,
                        result.start,
                        result.end,
                        result.code,
                        expr_list,
                    )
                    all_angelic_requests.append(request)

        logger.info("Angelic requests: {}", all_angelic_requests)

        return all_angelic_requests

    def _perform_angelic(
        self, test_content: str | None, thread: MessageThread
    ) -> AngelicTransaction | None:
        angelic_tx = None

        if config.disable_angelic:
            return None

        angelic_mode = AngelicMode.OFF

        at_least_one_loc_has_method = False
        for loc in self.bug_locs:
            if loc.method_name is not None:
                at_least_one_loc_has_method = True
                break

        # we can do imaginary as long as there is some method to reason about
        if self.bug_locs and at_least_one_loc_has_method:
            angelic_mode = AngelicMode.IMAGINARY

        # if there is reproducer test, we can upgrade to REAL mode
        if test_content is not None:
            angelic_mode = AngelicMode.REAL

        if angelic_mode == AngelicMode.OFF:
            return None

        tmp_thread = deepcopy(thread)
        tmp_thread.add_user(self._construct_angelic_select_prompt(angelic_mode))

        angelic_to_investigate_resp, *_ = common.SELECTED_MODEL.call(
            tmp_thread.to_msg(), response_format="json_object"
        )
        tmp_thread.add_model(angelic_to_investigate_resp)

        angelic_agent = AngelicAgent(self.task_dir, self.issue_stmt, test_content)

        try:
            requests = self._process_raw_angelic_request(angelic_to_investigate_resp)

            if requests:
                angelic_tx = angelic_agent.run(self.task, angelic_mode, requests)

        except Exception as e:
            logger.debug(
                "Exception happened during angelic request: {}, angelic-to-investigate response is {}",
                e,
                angelic_to_investigate_resp,
            )
            # dump tmp thread
            tmp_thread.save_to_file(
                Path(self.task_dir, "conv_angelic_tmp_thread_debug.json")
            )
            return None

        return angelic_tx

    def _write_patch(
        self,
        history_handles: list[PatchHandle] | None = None,
        test_content: str | None = None,
    ) -> tuple[bool, str, str, MessageThread, AngelicTransaction | None]:
        history_handles = history_handles or []

        thread = self._construct_init_thread()

        is_first_try = not any(handle in self._feedbacks for handle in history_handles)

        logger.debug(f"<agent write patch> is_first_try: {is_first_try}")

        for handle in history_handles:
            feedbacks = self._feedbacks.get(handle, [])
            if not feedbacks:
                logger.warning("patch {} does not have a feedback; skipping", handle)
                continue

            for tx in self._angelic_transactions[handle]:
                if tx.resp is None:
                    continue
                angelic_thread = self._reconstruct_angelic_thread(tx)
                thread.messages.extend(angelic_thread.messages)

            thread.add_model(self._responses[handle], [])

            for feedback in feedbacks:
                thread.add_user(feedback)

        angelic_tx = None

        # TODO: add the knob to turn on angelic here!
        if is_first_try:
            angelic_tx = self._perform_angelic(test_content, thread)

            if angelic_tx is not None and angelic_tx.resp is not None:
                # reconstruct and add this to the main patch-writing thread
                angelic_thread = self._reconstruct_angelic_thread(
                    angelic_tx, print=True
                )
                thread.messages.extend(angelic_thread.messages)

        thread.add_user(USER_PROMPT_INIT)

        if not history_handles:
            print_acr(USER_PROMPT_INIT)

        patch_resp, *_ = common.SELECTED_MODEL.call(thread.to_msg())
        thread.add_model(patch_resp)

        extract_status, _, diff_content = convert_response_to_diff(
            patch_resp, self.task_dir
        )
        record_extract_status(self.task_dir, extract_status)

        return (
            extract_status == ExtractStatus.APPLICABLE_PATCH,
            patch_resp,
            diff_content,
            thread,
            angelic_tx,
        )

    def _construct_init_thread(self) -> MessageThread:
        """
        Construct the initial patch gen conv thread, based on whether bug location is available.
        """
        if self.bug_locs:
            # bug location is available
            thread = MessageThread()
            thread.add_system(SYSTEM_PROMPT)
            thread.add_user(f"Here is the issue:\n{self.issue_stmt}")
            thread.add_user(self._construct_code_context_prompt())
        else:
            # bug location not there; we use the search conv history to at least get some context
            messages = deepcopy(self.context_thread.messages)
            thread = MessageThread(messages)
            thread = agent_common.replace_system_prompt(thread, SYSTEM_PROMPT)

        return thread

    def _construct_code_context_prompt(self) -> str:
        prompt = "Here are the possible buggy locations collected by someone else. "
        prompt += (
            "Each location contains the actual code snippet and the intended behavior of "
            "the code for resolving the issue.\n"
        )

        prompt += BugLocation.multiple_locs_to_str_for_model(self.bug_locs)
        prompt += (
            "Note that you DO NOT NEED to modify every location; you should think what changes "
            "are necessary for resolving the issue, and only propose those modifications."
        )
        return prompt

    def _reconstruct_angelic_thread(
        self, angelic_tx: AngelicTransaction, print=False
    ) -> MessageThread:
        thread = MessageThread()

        if angelic_tx.resp is None:
            # there was an error so there is no angelic response
            return thread

        angelic_select_prompt = self._construct_angelic_select_prompt(angelic_tx.mode)
        thread.add_user(angelic_select_prompt)

        if print:
            print_acr(angelic_select_prompt, "angelic value analysis")

        requests = angelic_tx.requests
        resp = angelic_tx.resp

        reformed_model_resp = [r.to_short_dict() for r in requests]
        model_resp_str = json.dumps(reformed_model_resp)
        thread.add_model(model_resp_str)

        # if print:
        #     print_angelic_select(model_resp_str)

        angelic_result = "Here is the result of the value-based analysis returned by the engineer:\n\n"
        if resp.buggy_trace is not None:
            angelic_result += "## Expression values in **buggy execution**:\n\n"
            angelic_result += resp.buggy_trace + "\n\n\n"
        if resp.angelic_trace is not None:
            angelic_result += (
                "## Expression values in the **correct/angelic execution**:\n\n"
            )
            angelic_result += resp.angelic_trace

        angelic_note = (
            "\nIMPORTANT: "
            "You SHOULD NOT completely believe the value-based analysis results, as they can be incorrect. "
            "You SHOULD take them only as a hint/reference to understand the bug better. "
            "You can also write a patch at a location that was not mentioned in the analysis, if you think it's better to patch there. "
            "You SHOULD still rely on your own understanding and reasoning to write a MINIMAL patch that resolves the issue. "
        )
        angelic_result += angelic_note

        if print:
            print_acr(angelic_result, "angelic value analysis result")

        thread.add_user(angelic_result)
        return thread

    def _register_applicable_patch(
        self, response: str, diff_content: str, angelic_txs: list[AngelicTransaction]
    ) -> PatchHandle:
        handle = str(self._request_idx)

        assert handle not in self._responses
        assert handle not in self._feedbacks
        assert handle not in self._diffs
        assert handle not in self._history

        self._responses[handle] = response
        self._diffs[handle] = diff_content
        self._angelic_transactions[handle] = angelic_txs
        self._history.append(handle)

        return handle


def generator(
    context_thread: MessageThread,
    output_dir: str,
) -> Generator[tuple[bool, str, str], str | None, None]:
    """
    Since the agent may not always write an applicable patch, we allow for retries.
    This is a wrapper around the actual run.

    Yields: is_success, result_message, patch_content
    """
    # (1) replace system prompt
    messages = deepcopy(context_thread.messages)
    new_thread: MessageThread = MessageThread(messages=messages)
    new_thread = agent_common.replace_system_prompt(new_thread, SYSTEM_PROMPT)

    # (2) add the initial user prompt
    new_thread.add_user(USER_PROMPT_INIT)
    print_acr(USER_PROMPT_INIT, "patch generation")

    index = 1
    while True:
        if index > 1:
            debug_file = pjoin(output_dir, f"debug_agent_write_patch_{index - 1}.json")
            new_thread.save_to_file(debug_file)

        logger.info(f"Trying to write a patch. Try {index}.")

        res_text, *_ = common.SELECTED_MODEL.call(new_thread.to_msg())

        new_thread.add_model(res_text, tools=[])
        print_patch_generation(res_text, f"try {index}")

        logger.info(f"Raw patch produced in try {index}. Writing patch into file.")

        raw_patch_file = pjoin(output_dir, f"agent_patch_raw_{index}")
        Path(raw_patch_file).write_text(res_text)

        # Attemp to extract a real patch from the raw patch
        with NamedTemporaryFile(prefix="extracted_patch-", suffix=".diff") as f:
            extract_status, extract_msg = extract_diff_one_instance(
                raw_patch_file, f.name
            )
            patch_content = Path(f.name).read_text()

        # record the extract status. This is for classifying the task at the end of workflow
        record_extract_status(output_dir, extract_status)

        if extract_status == ExtractStatus.APPLICABLE_PATCH:
            print_acr(f"```diff\n{patch_content}\n```", "extracted patch")

            validation_msg = yield True, "written an applicable patch", patch_content

            assert validation_msg is not None

            new_prompt = f"Your patch is invalid. {validation_msg}. Please try again:\n\n{USER_PROMPT_INIT}"
        else:
            _ = yield False, "failed to write an applicable patch", ""

            new_prompt = (
                "Your edit could not be applied to the program. "
                + extract_msg
                + " Please try again."
            )

        # TODO: we may not want to stick to a same thread, or the LLM may
        # be reluctant to try again.
        new_thread.add_user(new_prompt)
        print_patch_generation(new_prompt, f"feedback {index}")

        index += 1
