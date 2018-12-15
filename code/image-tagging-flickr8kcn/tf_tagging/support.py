import os
import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


def checkToSkip(filename, overwrite):
    to_skip = 0
    if os.path.exists(filename):
        if overwrite:
            logger.info("%s exists. overwrite", filename)
        else:
            logger.info("%s exists. skip", filename)
            to_skip = 1

    return to_skip


def makedirsforfile(filename):
    try:
        os.makedirs(os.path.split(filename)[0])
    except:
        pass


def niceNumber(v, maxdigit=6):
    """Nicely format a number, with a maximum of 6 digits."""
    assert(maxdigit >= 0)

    if maxdigit == 0:
        return "%.0f" % v

    fmt = '%%.%df' % maxdigit
    s = fmt % v

    if len(s) > maxdigit:
        return s.rstrip("0").rstrip(".")
    elif len(s) == 0:
        return "0"
    else:
        return s

