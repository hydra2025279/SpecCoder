import json
from dataclasses import dataclass
from enum import Enum
from os import PathLike


class AngelicMode(Enum):
    OFF = "OFF"
    REAL = "REAL"
    IMAGINARY = "IMAGINARY"


@dataclass
class AngelicRequest:
    file: str
    abs_file: str
    class_name: str | None
    method_name: str
    start: int
    end: int
    code: str
    expr_list: list[str]

    def __str__(self):
        return (
            f"<file>{self.file}</file>\n"
            f"<class_name>{self.class_name}</class_name>\n"
            f"<method_name>{self.method_name}</method_name>\n"
            f"<code>{self.code}</code>\n"
            f"<expressions>{self.expr_list}</expressions>\n"
        )

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "file": self.file,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "code": self.code,
            "expressions": self.expr_list,
        }

    def to_short_dict(self):
        return {
            "file": self.file,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "expressions": self.expr_list,
        }


@dataclass
class AngelicResponse:
    buggy_trace: str | None
    angelic_trace: str | None


@dataclass
class AngelicTransaction:
    requests: list[AngelicRequest]
    resp: AngelicResponse | None
    mode: AngelicMode

    def save_to_file(self, file: str | PathLike):
        content = "Mode:\n" + self.mode.value + "\n\n\n"

        requests = "\n".join([str(r) for r in self.requests])

        content += f"Request:\n{requests}\n\n\n"

        if self.resp is None:
            resp = ""
        else:
            resp = f"===Buggy Trace===\n{self.resp.buggy_trace}\n\n===Angelic Trace===\n{self.resp.angelic_trace}\n"
        content += "Response\n" + resp

        with open(file, "w") as f:
            f.write(content)

    def save_json_to_file(self, file: str | PathLike):
        content = {
            "mode": self.mode.value,
            "requests": [r.to_dict() for r in self.requests],
            "response": (
                None
                if self.resp is None
                else {
                    "buggy_trace": self.resp.buggy_trace,
                    "angelic_trace": self.resp.angelic_trace,
                }
            ),
        }

        with open(file, "w") as f:
            json.dump(content, f, indent=4)

    def __str__(self):
        return (
            f"Angelic Transaction:\nRequests: {self.requests}\nResponse: {self.resp}\n"
        )

    def __repr__(self):
        return self.__str__()
