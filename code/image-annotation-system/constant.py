import os
import logging


logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)

ROOT_PATH = os.path.join('sample_data') 
DATABASE_FILE = 'database/main.db'
VIS_FEAT = 'pyresnet152-pool5os'
SENT_COLLECTION = 'coco-translated-sentences'
IMG_COLLECTION = 'mscoco2014tags'
IMAGE_ROOT = '/imagedemo/' + IMG_COLLECTION + '-'
PAGE_LIMIT = 12
