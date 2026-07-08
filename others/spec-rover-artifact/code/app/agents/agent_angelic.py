from pathlib import Path

from loguru import logger

from app.angelic import driver as angelic_driver
from app.angelic.data_structures import (
    AngelicMode,
    AngelicRequest,
    AngelicResponse,
    AngelicTransaction,
)
from app.data_structures import MessageThread
from app.model import common
from app.task import SweTask, Task


class SystemPrompt:
    base = """You are a programming assistant which is expert in debugging. Someone has submitted
an issue report to your project, and your team is working on it to produce a patch for the issue.
Other engineers in your team have identified a few functions as buggy, and these are the potential fix locations.
They have also identified a few expressions in these functions that are likely to be related to the bug.
Your should reason about the values of these expressions in the buggy program, think about them,
and suggest how the values of these expressions should have changed if the bug was not present.

A buggy execution refers to an execution where the bug is present.
On the other hand, an angelic execution refers to an execution where the bug is fixed.
In an angelic execution, the bug in the program is fixed, and the program behaves as expected.

Your task is essentially to think about how the buggy exection can be turned in to an angelic execution.
"""

    common_notes = (
        "NOTE:\n"
        "- There may be multiple functions with multiple expressions for you to analyze. "
        "You SHOULD NOT attempt to analyze all the locations/expressions. "
        "Instead, focus on the important locations and expressison. "
        "You SHOULD make your own judgement, and only analyze the locations/expressions that are necessary for fixing the issue.\n"
        "- Your response should be precise and clear. DO NOT write lengthy responses.\n"
        "- The proposed changes to achieve the angelic execution should be MINIMAL.\n"
        "- When discussing about the executions, try to refer back to the line number of the code.\n"
    )

    real = (
        (
            base
            + """Now, another engineer has already written a reproducer test for the issue,
and we are executing the reproducer test on the buggy program to get a buggy execution trace.
You should observe the execution trace, see how the values of these expressions change during the execution,
and think about how the values should have changed if the bug was not present.
"""
        )
        + common_notes
    )

    imaginary = (
        (
            base
            + """You should relate the fix location functions back to the issue report,
and think about (1) in a buggy execution, what the values of the expressions should be, and
(2) if the issue was fixed, how the values of the expressions should have changed.
"""
        )
        + common_notes
    )


class AngelicAgent:
    def __init__(self, task_dir: str, issue_stmt: str, test_content: str | None):
        self.task_dir = task_dir
        self.issue_stmt = issue_stmt
        # The test to use for deriving buggy values and angelic values
        self.test_content: str | None = test_content

    def _construct_init_thread(self, mode: AngelicMode) -> MessageThread:
        assert mode != AngelicMode.OFF, "Angelic mode cannot be OFF."

        thread = MessageThread()
        if mode == AngelicMode.REAL:
            thread.add_system(SystemPrompt.real)
        elif mode == AngelicMode.IMAGINARY:
            thread.add_system(SystemPrompt.imaginary)

        thread.add_user(f"Here is the issue report:\n<issue>{self.issue_stmt}</issue>")
        return thread

    def _construct_bug_location_code_prompt(
        self, angelic_requests: list[AngelicRequest]
    ) -> str:
        prompt = "We are investigating these locations and expressions:\n"
        for idx, req in enumerate(angelic_requests):
            prompt += f"\n\nLocation {idx + 1} and expressions in it:\n"
            prompt += str(req) + "\n"

        return prompt

    def _construct_real_buggy_prompt(
        self, buggy_trace: str | None, angelic_requests: list[AngelicRequest]
    ) -> str:
        prompt = (
            "We have a reproducer test for this issue. "
            "We are executing the reproducer test on the buggy program. "
            "Below is a list of program locations (methods) together with some "
            "corresponding expressions. "
            "We want to observe that in the buggy execution, how the values of "
            "these expressions in the methods change.\n"
        )
        prompt += self._construct_bug_location_code_prompt(angelic_requests)
        prompt += "\n\n"

        if buggy_trace is not None:
            # have buggy trace, we can show it and ask for angelic trace
            prompt += "Here is the execution trace recording how the values of these expressions change during the excution:\n"
            prompt += buggy_trace
            prompt_for_angelic = (
                "Now think deeply about the buggy location code, the buggy execution trace, and the issue report.\n"
                "Discuss how the buggy execution trace can be turned in to an angelic execution.\n"
                "i.e discuss how the values of the expressions should have changed if the bug was not present.\n\n"
                "NOTE:\n"
                "- Your response should show what the values of the expressions SHOULD be, if the issue was not present or if the issue has been fixed.\n"
                "- The changed expression values in the angelic execution should give us a hint about the correct fix for the issue.\n"
            )
            prompt += prompt_for_angelic
        else:
            # Unfortuantely, we do not have the buggy trace.
            # This does not necessarily mean the execution did not reach the function entry of the location.
            # We should still prompt the model to at least get some angelic execution trace.
            prompt += (
                "The execution did not reach the function body of the bug location(s). "
                "You should still relate the buggy functions back to the issue report, "
                "and think about what the values of the expressions should be if the bug is fixed."
                "In your response, focus on discussing values of the expressions in an angelic execution."
            )

        prompt += (
            "\nIMPORTANT:\n"
            "- Your response should be precise and clear. DO NOT write lengthy responses. "
            "Instead, write short response, while focusing on the important expressions, at important "
            "program lines.\n"
            "- When giving the angelic execution trace, you MUST think about the MINIMAL changes required to fix the issue.\n"
        )
        return prompt

    def _construct_imaginary_buggy_prompt(
        self, angelic_requests: list[AngelicRequest]
    ) -> str:
        prompt = (
            "Below is the list of program locations (methods) together with some "
            "corresponding expressions. "
        )
        prompt += self._construct_bug_location_code_prompt(angelic_requests)
        prompt += "\n\n"

        prompt += (
            "Now relate these functions back to the issue report. Think about "
            "(1) in a buggy execution, what the values of the expressions are, and "
            "(2) if the issue was fixed (i.e. angelic execution), how the values of the expressions should have changed.\n\n"
            "Write your response in the following format:\n"
            "# Buggy exection\n"
            "... describe the values of the expressions in the buggy execution ...\n"
            "# Angelic execution\n"
            "... describe the values of the expressions in the angelic execution ...\n\n"
            "You MUST follow this format strictly in your response. DO NOT write anything extra."
            "IMPORTANT:\n"
            "- Your response should be precise and clear. DO NOT write lengthy responses. "
            "Instead, write short response, while focusing on the important expressions, at important "
            "program lines.\n"
            "- When giving the angelic execution trace, you MUST think about the MINIMAL changes required to fix the issue.\n"
            "- When there are multiple locations and expressions to analyze, you SHOULD NOT attempt to analyze all of them. "
            "Instead, think about which locations/expressions are important to the bug/issue, and focus on those.\n"
        )

        return prompt

    def run_real_mode(
        self,
        task: SweTask,
        angelic_requests: list[AngelicRequest],
        thread: MessageThread,
    ) -> tuple[AngelicResponse | None, MessageThread]:
        # run buggy program with reproducer to get buggy trace
        buggy_trace: str | None = self.get_buggy_trace(task, angelic_requests)

        # If buggy trace is None, since we are already in the real mode, still let
        # the model reason about angelic execution. The prompt to catch this is in
        # `_construct_real_buggy_prompt`.

        # if buggy_trace is None:
        #     buggy_func_names = [req.location.method_name for req in angelic_requests]

        #     buggy_trace = f"Reproducer execution did not reach the function body of the methods {buggy_func_names}. "
        #     buggy_trace += "Either the bug happened at the function entry, or the function is not on the execution path."

        # construct a prompt and send this buggy trace to model
        thread.add_user(
            self._construct_real_buggy_prompt(buggy_trace, angelic_requests)
        )

        thread.save_to_file(Path(self.task_dir) / "conv_angelic_REAL_before.json")

        try:
            response, *_ = common.SELECTED_MODEL.call(thread.to_msg())
        except Exception:
            return None, thread

        # print(f"Angelic response content: {response}")

        thread.add_model(response)
        thread.save_to_file(Path(self.task_dir) / "conv_angelic_REAL_after.json")

        angelic_response = AngelicResponse(buggy_trace, response)
        return angelic_response, thread

    def run_imaginary_mode(
        self,
        angelic_requests: list[AngelicRequest],
        thread: MessageThread,
    ) -> tuple[AngelicResponse | None, MessageThread]:
        thread.add_user(self._construct_imaginary_buggy_prompt(angelic_requests))

        thread.save_to_file(Path(self.task_dir) / "conv_angelic_IMAGINARY_before.json")

        try:
            response, *_ = common.SELECTED_MODEL.call(
                thread.to_msg(),
            )
        except Exception:
            return None, thread

        thread.add_model(response)
        thread.save_to_file(Path(self.task_dir) / "conv_angelic_IMAGINARY_after.json")

        # parse the response
        seperator = "# Angelic execution"
        seperator_idx = response.find(seperator)

        if seperator_idx == -1:
            # not found => response is malformed
            return None, thread

        buggy_trace = response[:seperator_idx]
        angelic_trace = response[seperator_idx:]

        angelic_response = AngelicResponse(buggy_trace, angelic_trace)
        return angelic_response, thread

    def run(
        self, task: Task, mode: AngelicMode, angelic_requests: list[AngelicRequest]
    ) -> AngelicTransaction | None:
        """
        Returns:
            - Completed transaction
            - None if failed to complete the transaction
        """
        assert isinstance(
            task, SweTask
        ), "Angelic values are only supported in SweTask."

        assert mode != AngelicMode.OFF, "Angelic mode cannot be OFF."

        thread = self._construct_init_thread(mode)

        angelic_resp = None
        if mode == AngelicMode.REAL:
            logger.info("<agent angelic> Performing angelic analysis in REAL mode.")
            angelic_resp, _ = self.run_real_mode(task, angelic_requests, thread)
        else:  # mode == AngelicMode.IMAGINARY:
            logger.info(
                "<agent angelic> Performing angelic analysis in IMAGINARY mode."
            )
            angelic_resp, _ = self.run_imaginary_mode(angelic_requests, thread)

        if angelic_resp is None:
            logger.info(
                "<agent angelic> Angelic resp is None; returning None as AngelicTransaction."
            )
            return None

        angelic_tx = AngelicTransaction(angelic_requests, angelic_resp, mode)
        logger.info(f"<agent angelic> Angelic transaction ok: {angelic_tx}")

        return angelic_tx

    def get_buggy_trace(
        self, task: SweTask, angelic_requests: list[AngelicRequest]
    ) -> str | None:
        """
        Returns:
            None if the execution did not reach the body of the buggy function,
            or something went wrong during the execution.
        """

        assert isinstance(
            task, SweTask
        ), "Angelic values are only supported in SweTask now."

        assert self.test_content is not None, "None test content in real mode"
        instru_output = angelic_driver.instrument_and_trace(
            task, self.test_content, angelic_requests
        )
        return instru_output
