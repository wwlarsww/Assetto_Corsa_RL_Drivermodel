"""Change test settings independently of the user's template values/comments."""

import re


def replace_setting(content: str, key: str, assignment: str | None, *, section: str | None = None) -> str:
    start, end = 0, len(content)
    if section is not None:
        header = re.search(rf"(?m)^\[{re.escape(section)}\]\s*\n", content)
        assert header is not None, section
        start = header.end()
        next_header = re.search(r"(?m)^\[", content[start:])
        if next_header is not None:
            end = start + next_header.start()
    replacement = "" if assignment is None else assignment + "\n"
    body, count = re.subn(rf"(?m)^{re.escape(key)}\s*=[^\n]*(?:\n|$)",
                         lambda _: replacement, content[start:end], count=1)
    assert count == 1, key
    return content[:start] + body + content[end:]
