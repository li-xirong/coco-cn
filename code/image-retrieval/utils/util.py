import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

def check_img_list(sentid, imgid):
    true_imgid = sentid.split('#')[0]
    if true_imgid.endswith('.jpg') or true_imgid.endswith('.mp4'):
        true_imgid = true_imgid[:-4]
    if  true_imgid== imgid:
        return 1
    else:
        return 0


def readSentsInfo(inputfile):
    sent_ids = []
    sents = []
    id2sents = {}
    for line in open(inputfile):
        data = line.strip().split(' ', 1)
        sent_ids.append(data[0])
        sents.append(data[1])
        id2sents[data[0]] = data[1]
    return (sent_ids, sents, id2sents)


def load_config(config_path):
    variables = {}
    exec(compile(open(config_path, "rb").read(), config_path, 'exec'), variables)
    return variables['config']


def which_language(collection):
    lang = 'zh' if collection.find('cn')>0 else 'en'
    logger.info('check language of %s -> %s', collection, lang)
    return lang

if __name__ == '__main__':
    for collection in 'cococntrain'.split():
        which_language(collection)

