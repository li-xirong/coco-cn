import web
import os
import json

config = json.load(open('config.json'))
imagedata_path = config['imagedata_path']
trainCollection = config['trainCollection']
testCollection = config['testCollection']


cType = {"png":"images/png",
         "jpg":"images/jpeg",
         "jpeg":"images/jpeg",
         "gif":"images/gif",
         "ico":"images/x-icon"}

def im2path(collection, name, big=True):
    img_dir = os.path.join(imagedata_path, collection, 'ImageData128x128')
    img_dir = img_dir.replace('128x128','') if big else img_dir
    ext = '.JPEG' if collection.startswith('imagenet') else '.jpg'
    if collection.startswith('flickr30k'):
        img_dir = os.path.join(img_dir, os.path.splitext(name)[0][-1])
    return os.path.join(img_dir, os.path.splitext(name)[0] + ext)


class images:
    def get_local(self, name):
        return im2path(trainCollection, name, True)    

    def GET(self,name):
        ext = name.split(".")[-1] # Gather extension
        if name.find('.')<0:
            ext = 'jpg'
        imfile = self.get_local( os.path.splitext(name)[0] )
        #print imfile
        try:
            web.header("Content-Type", cType[ext]) # Set the Header
            return open(imfile,"rb").read() # Notice 'rb' for reading images
        except:
            raise web.notfound()

class queryimages (images):
    def get_local(self, name):
        return im2path(testCollection, name, True)
            
class bigimages (images):
    def get_local(self, name):
        return im2path(name, True)

            
if __name__ == '__main__':
    im = bigimages()
    im.GET('4362444639.jpg')

