#Processes data from NIST Special Database 9
#
#Provides function to load and resize data. 
#It also includes a Keras Data Generator for training networks on this data.
#The data generator allows generated data to be random crops of a fixed size of the orginal data.
#
#Philip Bontrager
#Written for Python 3.5 and Keras 1.1.2
#December 13, 2016


from keras.preprocessing.image import *
from PIL import Image
import numpy as np
import os
from natsort import natsorted, ns
# Files
from thumb_from_sd09 import scan_dir


def label_sd09(class_name):
  """
  Create a unique label from a file name,
  according to the NIST SD09 naming scheme
  """
  code = class_name.split('_')[1]
  num = code[:-1]
  character = code[-1]
  label = 2 * int(num) - (character == 'f')
  return label

def load_data(s = 256, dirname = os.getcwd()):
    imgs_names = scan_dir(dirname, ".png")
    # Sort names alphanumerically
    imgs_names = natsorted(imgs_names, alg=ns.PATH)
    
    #Prep Images
    X = []
    Y = []
    for label, i in enumerate(imgs_names):
        im = Image.open(i)
        if(im.size[0] >= s and im.size[1] > s):
            Y.append(label)
            X.append(resize(im, s))
    
    X_train = np.array(X)
    Y_train = np.array(Y)
    
    return X_train, Y_train

def resize(img, s):
    if(img.size[0] > img.size[1]):
        h = s
        ratio = h/float(img.size[1])
        w = int(img.size[0]*ratio)
        img = img.resize((w,h), Image.ANTIALIAS)
        
        border = int((w - s)/2)
        img = img.crop((border, 0, border + s, h))
    else:
        w = s
        ratio = w/float(img.size[0])
        h = int(img.size[1]*ratio)
        img = img.resize((w,h), Image.ANTIALIAS)
        
        border = int((h - s)/2)
        img = img.crop((0, border, w, border + s))
        
    img = img.convert('L')    
    #img = preK.img_to_array(img, 'tf')
    img = img_to_array(img)
    
    return img
    

#Added a random crop to the random_transform method
#preK infront of both ImageDataGenerators
class DataGenerator(ImageDataGenerator):

    def __init__(self, crop_size = 32, 
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=False,
                 preprocessing_function=None,
                 data_format=K.image_data_format()):
        ImageDataGenerator.__init__(self, featurewise_center=featurewise_center,
                                    samplewise_center=samplewise_center,
                                    featurewise_std_normalization=featurewise_std_normalization,
                                    samplewise_std_normalization=samplewise_std_normalization,
                                    zca_whitening=zca_whitening,
                                    zca_epsilon=zca_epsilon,
                                    rotation_range=rotation_range,
                                    width_shift_range=width_shift_range,
                                    height_shift_range=height_shift_range,
                                    shear_range=shear_range,
                                    zoom_range=zoom_range,
                                    channel_shift_range=channel_shift_range,
                                    fill_mode=fill_mode,
                                    cval=cval,
                                    horizontal_flip=horizontal_flip,
                                    vertical_flip=vertical_flip, 
                                    rescale=rescale,
                                    preprocessing_function=preprocessing_function,
                                    data_format=data_format)
        self.crop_size = crop_size
        
    def random_crop(self, img):
        x, y = (np.random.randint(0, img.shape[1] - self.crop_size) for i in range(2))
        return img[x:x + self.crop_size, y:y + self.crop_size, :] # (h,w,c)
#        return img[:, x:x + self.crop_size, y:y + self.crop_size]
        
    def flow_random(self, X, y=None, batch_size=32, shuffle=True, seed=None,
        save_to_dir=None, save_prefix='', save_format='jpeg'):
            
        return CropArrayIterator(X, y, self,batch_size=batch_size, 
            crop_size=self.crop_size, shuffle=shuffle, seed=seed,
            data_format=self.data_format, save_to_dir=save_to_dir, 
            save_prefix=save_prefix, save_format=save_format)

    
class CropArrayIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, crop_size=32, shuffle=False, seed=None,
                 data_format=K.image_data_format(),
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if y is not None and len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        self.X = X
        self.y = y
        self.crop_size = crop_size
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(CropArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)
        
    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array = next(self.index_generator)
#            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        #batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        current_batch_size = len(index_array)
        current_index = self.batch_index
        batch_x = np.zeros((current_batch_size, self.crop_size, self.crop_size, self.X.shape[-1]))
        for i, j in enumerate(index_array): 
            x = self.image_data_generator.random_crop(self.X[j])
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y