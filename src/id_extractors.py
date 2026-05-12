import re
from typing import Optional


def numeric_filename_extractor(
    filename: str
) -> Optional[int]:

    """
    Extract first numeric ID from filename.

    Examples:
        person001.jpg → 1
        user_42.png → 42
    """

    match = re.search(r"(\d+)", filename)

    if match:
        return int(match.group(1))

    return None


def custom_regex_extractor(
    pattern: str
):

    regex = re.compile(
        pattern,
        re.IGNORECASE
    )

    def extractor(filename: str):

        match = regex.match(filename)

        if match:
            return int(match.group(1))

        return None

    return extractor