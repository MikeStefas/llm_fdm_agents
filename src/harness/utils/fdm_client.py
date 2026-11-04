from typing import Any
from typesafe_sdk import TypeSafeClient
from typing_extensions import override


class DeterministicFDMClient(TypeSafeClient):

    @override
    def system_one(self, state: Any, questions: Any) -> Any:
        return super().system_one(
            state=state,
            questions=questions,
            extra_body={"temperature": 0.0, "seed": 42},
        )

