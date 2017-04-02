def read_ims(directory, imsz, whitening=False):
  ''' Reads in images in subdirectories located in directory and 
      assigns a unique one-hot vector to each image in the respective
      folder.
      
      args:
           directory: the location of all the folders containing
                      each image class.
           imsz: resizes the width and height of each image to 
                 imsz
           whiten: Whitens images. Default is no whitening. Images
                   must be grayscale. 
           save: saves the images and labels as an h5 file. Arg is
                 a list with three strings containing the key for the 
                 data and the key for the labels. For example, 
                 ['images_labels.h5', 'images', 'labels']. 
                 Defaults to no saving. '''
 
  main_dir=os.getcwd()
  os.chdir(directory)
  if whitening is True:
    num_channels=1
  else:
    num_channels=3
  num_ims=sum([len(files) for r, d, files in os.walk(directory)])
  imgs=np.zeros([num_ims, imsz, imsz, num_channels])
  labels=np.zeros([num_ims, len(os.listdir(os.getcwd()))])
  im_num=0  

  for f in os.listdir(os.getcwd()):
    if os.path.isdir(f):
        print('Folder name: %s'%(f))
        os.chdir(f)
        r0=np.argmin(np.sum(labels, axis=1))
        c0=np.argmin(np.sum(labels, axis=0))
        labels[r0:r0+len(glob.glob1(os.getcwd(), '*')), c0]=1

        for filename in os.listdir(os.getcwd()):
          #print(filename)
          im=imresize(imread(filename), [imsz, imsz])
          if whitening is True:
            im=whiten(scale(im, axis=0, with_mean=True, with_std=True, copy=True))
            im=im[:, :, np.newaxis]
          imgs[im_num, :, :, :]=im
          if im.shape[2]!=num_channels:
            print('Check %s file, wrong size'%(filename))
            sys.exit(0)
          im_num+=1
        os.chdir(directory)
  os.chdir(main_dir)
  return imgs, labels
