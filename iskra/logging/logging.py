# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import logging

from rich.logging import RichHandler


def getLogger(name: str) -> logging.Logger:  # noqa: N802
    logger = logging.getLogger(name)
    handler = RichHandler(rich_tracebacks=True, markup=True)
    format_str = "[bold][red]%(name)-12s[/bold][/red] %(message)s"
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    return logger
