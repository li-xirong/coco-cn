from __future__ import print_function
import web
import time

import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)
logger.setLevel(logging.INFO)

import constant
import userControl
import sentence
import image
from imagedemo import imagedemo

import sys
reload(sys)
sys.setdefaultencoding('utf8')

web.config.debug = False

render = web.template.render('templates/')
urls = (
    '/', 'index',
    '/image', 'image',
    '/history', 'history',
    '/imagedemo/(.*)', 'imagedemo'
)

app = web.application(urls, globals())
session = web.session.Session(app, web.session.DiskStore('sessions'), initializer={'login': 0, 'user': 0})

user_control = userControl.user(constant.DATABASE_FILE)
sen = sentence.sentence(constant.DATABASE_FILE)
image_pool = image.image(constant.DATABASE_FILE)


def check():
    if session.login == 0:
        judge = 4
        resp = {'judge': judge}
        return True, render.index(resp)
    else:
        return False, None


class index():
    def GET(self):
        judge = 4
        resp = {'judge': judge}
        return render.index(resp)

    def POST(self):
        i = web.input()
        username = i.get('user_id')
        password = i.get('password')

        judge = user_control.checkUser(username, password)

        if judge == 1:
            session.user = user_control.getUserId(username)
            session.login = 1
            raise web.redirect('/image')
        else:
            resp = {'judge': judge}
            return render.index(resp)


class image:
    def GET(self):
        j, r = check()
        if j:
            return r

        i = web.input()
        s_sen = ''
        s_label = ''
        rank = 0
        img_id = i.get('image_id')
        is_edit = 0
        if img_id is not None:
            is_edit = 1
            image_id = int(img_id)
        else:
            # get a img id from img pool
            image_id = image_pool.getimageid(session.user)
        j, imageid, image_id = user_control.getimageid(session.user, image_id)
        if j:
            is_edit = 1
        rank, s_sen, s_label = sen.get_sentence_by_imageid(session.user, image_id)
        image_name = image_pool.getimagename(imageid)
        label = ', '.join(filter(lambda x: x, image_pool.getlabel(imageid).split(' ')))
        logger.info(label)

        image_url = constant.IMAGE_ROOT + image_name

        sentences = sen.getSentence(imageid)
        logger.info("s_sen: %s", s_sen.decode('gbk'))

        number = sen.getNumber(session.user)

        logger.info(s_label)

        resp = {'image_id': image_id, 'image': image_url, 'sentence': sentences, 'label': label,
                'submitted_sentence': s_sen, 'submitted_label': s_label, 'rank': rank, 'number': number,
                'is_edit': is_edit, 'judge': j}
        return render.image(resp)

    def POST(self):
        i = web.input()

        image_id = int(i.get('image_id'))
        submitted_sentence = i.get('sentence').encode('gbk')
        selected = int(i.get('selected'))
        original_sentence = i.get('selectedSentence').encode('gbk')
        labels = i.get('label')
        edit = i.get('edit')

        j, real_id, image_id = user_control.getimageid(session.user, image_id)
        sen.save_sentence(session.user, image_id, time.time(), original_sentence, selected, submitted_sentence, labels, real_id)
        image_pool.save(image_id)

        logger.info(edit)
        if int(edit) == 0:
            user_control.update_cursor(session.user)

        return True


class history:
    def GET(self):
        j, r = check()
        if j:
            return r

        i = web.input()
        page = i.get("page")
        if page is None:
            page = 1

        judge, data, page_number = sen.get_sentence(session.user, int(page))
        resp = {'judge': judge, 'data': data, 'page': page_number, 'current': page}

        return render.history(resp)


if __name__ == "__main__":
    app.run()
