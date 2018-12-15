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


