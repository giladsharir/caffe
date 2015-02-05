# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
show_fig=False

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    plt.show()
    
    
def draw_grad(net):
    
    print "gradient plot"
    g_a=np.zeros((3,227,227))
    d_a=np.zeros((3,227,227))
#     for i in range(10):
    net.forward()
    net.backward()
    
    p = net.blobs['fc8'].data[0]
    lbl = np.argmax(p)
    print "label: {}".format(lbl)
    
    c = net.blobs['data'].diff
    d = net.blobs['data'].data
    
    g = abs(c[0,:,:,:]).max(axis=0)
        
#         g = g+c[0,:,:,:]
#     img = np.zeros((g.shape[0],g.shape[1],3))
#     for i in range(3):
#         img[:,:,i] = d[0,i,:,:].astype('uint8')
    
    if show_fig: 
        plt.subplot(2,1,1); plt.imshow(g,cmap = cm.Greys_r); plt.subplot(2,1,2); plt.imshow(d[0,0,:,:],cmap = cm.Greys_r);plt.show() 
        
        #plt.imshow(c[0,0,:,:],cmap = cm.Greys_r)
        
    return g,lbl
    
def run_seg(img_,g):
    
    #img_ = plt.imread('images/8bd1f0a8-2b01-4322-b88f-b54223f12d4b.jpg')
    #img_ = cv2.resize(img_,g.shape)
    img_ = cv2.resize(img_,(256,256))
    d = int(np.floor((256-g.shape[1])/2))
    try:
        img_ = img_[d:256-d-1,d:256-d-1,:]
    except:
        img_ = img_[d:256-d-1,d:256-d-1]
#     img_ = cv2.resize(img_,g.shape)
#     TH_high = np.percentile(g, 97)
    TH_high = np.percentile(g, 99)
    TH_low = np.percentile(g, 5)
    mask = np.zeros(g.shape).astype('uint8')
    mask[g>TH_low]=2
    mask[g>TH_high]=1
    tmp1 = np.zeros((1,65),np.float64)
    tmp2 = np.zeros((1,65),np.float64)    
    try: 
        cv2.grabCut(img_,mask,None,tmp1,tmp2,10,mode=cv2.GC_INIT_WITH_MASK)
    except:
        #mask = np.zeros(g.shape).astype('uint8')
        pass
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img_*mask2[:,:,np.newaxis]
    
    if show_fig:
        plt.imshow(img),plt.show()
    
    open_img = ndimage.binary_opening(mask2)
    close_img = ndimage.binary_closing(open_img)
    label_im, nb_labels = ndimage.label(close_img)
    
    s = np.asarray([sum(sum(label_im==i)) for i in range(1,nb_labels+1)])
    mask_=(label_im == s.argmax()+1)
    img = img_*mask_[:,:,np.newaxis]
    
    if show_fig:
        plt.imshow(img),plt.show()
        plt.imshow(img_); plt.contour(mask_, [0.5], linewidths=2, colors='r'); plt.axis('off'); plt.show()
        
        
        
        
    
    return mask_
    
    
    
    
    
    
def feature_compute(im_name,net):
    try:
        im = caffe.io.load_image(im_name)
    except:
        print "error reading {}".format(im_name)
        pass
     
    #out2 = net.forward_all(data=np.asarray([net.preprocess('data', im)]))
    
    #o = net.forward_backward_all(data=np.asarray([net.preprocess('data', im)]))
    img = plt.imread(im_name)
    g,lbl = draw_grad(net)
    mask = run_seg(img,g)
    
    save_dir = '/home/gilad/Devel/caffe/python/results/gradient_ascent_getty/cat/masks'
    p1 = im_name.rfind('.')
    p2 = im_name.rfind('/')
    mask_name = save_dir + im_name[p2:p1]+'_mask_' +str(lbl)+'.jpg'
    
    plt.imsave(mask_name,mask)
    
    #c = net.blobs['conv3'].diff
#     li = open('Catalog_list.txt')
#     for i in range(5):
#         name = li.readline()
#         name = name[:name.find(' ')].strip()
#         img = plt.imread(name)
#         g = draw_grad(net)
#         run_seg(img,g)
#     li.close()
        
    
    # show net input and confidence map (probability of the top prediction at each location)
    '''
    plt.subplot(1, 2, 1)
    plt.imshow(net_full_conv.deprocess('data', net_full_conv.blobs['data'].data[0]))
    plt.subplot(1, 2, 2)
    plt.imshow(out['prob'][0].max(axis=0))
    plt.show()

    
    feats = net_full_conv.blobs['fc6-conv'].data[0,:81]
    vis_square(feats,padval=1)
    '''    
    
    #net_full_conv.blobs['fc6-conv'].data.shape
    #use fc6-conv layer for features:
    ind_max = np.argmax(abs(net.blobs['fc6'].data[0]))
    ind_sort = (-np.array(net.blobs['fc6'].data[0]).squeeze()).argsort()[:100]
#     feat = net_full_conv.blobs['fc6-conv'].data[0,ind_sort,:,:].reshape(1,64*len(ind_sort)).squeeze()
    #feat = net_full_conv.blobs['conv2'].data.reshape(1,1,1,55**2 * 256).squeeze()
#     feat_ = feat
#     if sum(np.power(feat,2))!=0:
#         feat_ = np.divide(feat, np.sqrt(sum(np.power(feat,2))))
    
    #print sum(feat**2)
#     return feat_.copy()

def draw_confusion_table(features):
   
    
    N = len(features)
    #IM = [[0]*N]*N
    a = np.zeros((N,N))
    
    
    for i in range(0,N):
        for j in range(0,N):
            a[i,j] = np.dot(features[i],features[j])
            #a[i,j] = np.sqrt(sum(np.power(features[i]-features[j],2)))
            if i==j:
                a[i,j]=0.0
            #a[i,j] = sum(abs(np.dot(features[i],features[j])))
            

    m = np.array(a)
    b = np.tile(np.amax(m,1),(len(m[0]),1)).reshape(len(m[0]),len(m)).transpose()
    IM0 = np.divide(m,b)
    IM = np.zeros((N+1,N+1))
    IM[1:,1:]=IM0
                
    plt.imshow(IM,interpolation='none')
    plt.axis([0.5, N+0.5, 0.5, N+0.5])
    
    plt.show()
            
    #features = feat/max(feat)
    '''
    N = len(features)
    for i in range(1,N):
        
        plt.subplot(N,N,i)
        r=int(i)//int(N)
        c=i%N
        val = sum(abs(features[c]-features[r]))
        IM = np.ones(10, np.uint8)*(val)
        plt.imshow(IM)
        plt.show()
     '''   
                
def main(argv):
        # Make sure that caffe is on the python path:
    caffe_root = '/home/gilad/Devel/caffe/'  # this file is expected to be in {caffe_root}/examples
    import sys
    import glob
    sys.path.insert(0, caffe_root + 'python')
    
    
    
    # Load the original network and extract the fully-connected layers' parameters.
    #net_untrained = caffe.Net(caffe_root+'models/bvlc_reference_caffenet/deploy.prototxt')
    
    #net = caffe.Net(caffe_root+'models/bvlc_reference_caffenet/deploy.prototxt', caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    net = caffe.Net(caffe_root+'models/bvlc_reference_caffenet/train_val_index.prototxt', caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    #net = net_untrained
    params = ['fc6', 'fc7', 'fc8']
    # fc_params = {name: (weights, biases)}
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
    
    net.set_phase_test()
    net.set_mode_cpu()
    
    for fc in params:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)
    
    #%matplotlib inline
    
    # load input and configure preprocessing
#     feat=[]
    #for i in range(1,19):
#   for im_name in glob.glob("/home/gilad/tmp/*.jpg"):    
    li = open('Catalog_list.txt')
    for name in li.readlines():
#         name = li.readline()
        im_name = name[:name.find(' ')].strip()   
        feature_compute(im_name, net)
        #im_name = caffe_root+'examples/images/bike/bike'+str(i)+'.jpg'
#         feat.append(feature_compute(im_name, net_full_conv, net))
        
#     draw_confusion_table(feat)
        
    #print sum(abs(feat))    
'''        

    im_name = caffe_root+'examples/images/cat.jpg'
    feat1 = feature_compute(im_name,net_full_conv)
    
    im_name = caffe_root+'examples/images/cat2.jpg'
    feat2 = feature_compute(im_name,net_full_conv)

    im_name = caffe_root+'examples/images/cat3.jpg'
    feat3 = feature_compute(im_name,net_full_conv)
    
    im_name = caffe_root+'examples/images/cat4.jpg'
    feat4 = feature_compute(im_name,net_full_conv)
    
    im_name = caffe_root+'examples/images/cat5.jpg'
    feat5 = feature_compute(im_name,net_full_conv)
    
    im_name = caffe_root+'examples/images/cat6.jpg'
    feat6 = feature_compute(im_name,net_full_conv)
    
    im_name = caffe_root+'examples/images/cat7.jpg'
    feat7 = feature_compute(im_name,net_full_conv)
    
    im_name = caffe_root+'examples/images/goldfish.jpeg'
    feat8 = feature_compute(im_name,net_full_conv)
'''
    
    #print sum(abs(feat1[:]-feat2[:]))
    #print sum(abs(feat1-feat3))
    #feat1-feat2
    
    
#convert to full conv model and extract features
if __name__ == "__main__":
    import sys
    import caffe
    import numpy as np
    import matplotlib.pyplot as plt 
    import matplotlib.cm as cm
    import cv2
    from scipy import ndimage
    
    main(sys.argv)
        

