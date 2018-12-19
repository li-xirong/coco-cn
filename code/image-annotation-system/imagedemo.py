from __future__ import print_function

import os
import web

from constant import ROOT_PATH

import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)
logger.setLevel(logging.INFO)


cType = {".png":"images/png",
         ".jpg":"images/jpeg",
         ".jpeg":"images/jpeg",
         ".gif":"images/gif",
         ".ico":"images/x-icon"}


class imagedemo:
    def get_local(self, name):
        collection, image_id = name.split('-')
        image_id = image_id if image_id.endswith('.jpg') else image_id+'.jpg'
        return os.path.join(ROOT_PATH, collection, 'ImageData', image_id)

    def GET(self,name):
        imfile = self.get_local(name)
        logger.info(imfile)
        try:
            web.header("Content-Type", cType[os.path.splitext(imfile)[1]]) # Set the Header
            return open(imfile,"rb").read() # Notice 'rb' for reading images
        except Exception, e:
            logger.error(e)
            raise web.notfound()
