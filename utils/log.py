import logging
from rich.logging import RichHandler
from rich.console import Console
from typing import Optional


def setup_logger(
    level: str = "INFO", logfile: Optional[str] = None, time: Optional[bool] = False
) -> logging.Logger:
    """Setup a logger that uses RichHandler to write the same message both in stdout
    and in a log file called logfile. Level of information can be customized and
    dumping a logfile is optional.

    :param level: level of information
    :type level: str, optional
    :param logfile: file where information are stored
    :type logfile: str
    """
    logger = logging.getLogger()
    #    __name__
    #)  # need to give it a name, otherwise *way* too much info gets printed out from e.g. numba

    # Set up level of information
    possible_levels = ["INFO", "DEBUG"]
    if level not in possible_levels:
        raise ValueError(
            "Passed wrong level for the logger. Allowed levels are: {}".format(
                ", ".join(possible_levels)
            )
        )
    logger.setLevel(getattr(logging, level))

    formatter = logging.Formatter("%(message)s")
    if time:
        formatter = logging.Formatter("%(asctime)s %(message)s")

    # Set up stream handler (for stdout)
    stream_handler = RichHandler(show_time=False, rich_tracebacks=True)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Set up file handler (for logfile)
    if logfile:
        file_handler = RichHandler(
            show_time=False,
            rich_tracebacks=True,
            console=Console(file=open(logfile, "wt")),
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Silence Dask logging
    dask_logger = logging.getLogger('distributed')
    # Set its level to an 'impossible' level so no logs from this logger will propagate
    dask_logger.setLevel(logging.CRITICAL + 1)

    return logger