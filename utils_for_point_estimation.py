# -*- coding: utf-8 -*-
#一貫して画像の読み込み,操作にはcv2を使用します。なのでchianer用に変換する関数もあります。
#その一方で回転、スライドの処理などはchainercvに既存関数があり、便利なのでそれを使います。
#ですので、一貫性を保つため、chainercvで操作をした後にcv2用に軸を逐一動かす方針にします。
#また、座標を示す際に表記ゆれがあると処理がだいぶ面倒なので、以下のように統一します：
#[y0,x0,y1,x1,y2,x2,y3,x3,.....,yn,xn]
import json
import numpy as np
import cv2
import ast
import chainercv
import os
import time
import sys
import datetime
import random
import time

output_dir_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def draw_marker(img,points,Size=20,Thickness=1,color=(255,255,255)):
    result = img.copy()
    point = np.array(points).reshape([3,2])
    for y,x in point:
        cv2.drawMarker(result,(x,y),color,markerType=cv2.MARKER_CROSS, markerSize=Size, thickness=Thickness, line_type=cv2.LINE_8)
    return result


#make mask for grabcut
#input:a image(to refer width and height)
#output:a mask image(gray-scale)
def make_mask(img):
    h,w = img.shape[:2]
    img_tmp = np.full((h,w),100).astype('uint8')

    img_tmp = cv2.circle(img_tmp, center=(w // 2,h // 2), radius=80, color=3, thickness=-1)
    img_tmp = cv2.circle(img_tmp, center=(w // 2,h // 2), radius=45, color=255, thickness=-1)

    img_tmp = cv2.circle(img_tmp, center=(0,h // 2), radius=40, color=2, thickness=-1)
    img_tmp = cv2.circle(img_tmp, center=(w,h // 2), radius=40, color=2, thickness=-1)

    img_tmp = cv2.circle(img_tmp, center=(0+20,0+20), radius=60, color=2, thickness=-1)
    img_tmp = cv2.circle(img_tmp, center=(0+20,0+20), radius=20, color=0, thickness=-1)

    img_tmp = cv2.circle(img_tmp, center=(w-20,h-20), radius=60, color=2, thickness=-1)
    img_tmp = cv2.circle(img_tmp, center=(w-20,h-20), radius=20, color=0, thickness=-1)

    img_tmp = cv2.circle(img_tmp, center=(0+20,h-20), radius=60, color=2, thickness=-1)
    img_tmp = cv2.circle(img_tmp, center=(0+20,h-20), radius=20, color=0, thickness=-1)

    img_tmp = cv2.circle(img_tmp, center=(w-20,0+20), radius=60, color=2, thickness=-1)
    img_tmp = cv2.circle(img_tmp, center=(w-20,0+20), radius=20, color=0, thickness=-1)

    img_tmp = cv2.rectangle(img_tmp,(0,0),(45,h),2,-1)
    img_tmp = cv2.rectangle(img_tmp,(0,h-45),(w,h),2,-1)
    img_tmp = cv2.rectangle(img_tmp,(w-45,0),(w,h),2,-1)
    img_tmp = cv2.rectangle(img_tmp,(0,0),(w,45),2,-1)

    img_tmp = cv2.rectangle(img_tmp,(0,0),(10,h),0,-1)
    img_tmp = cv2.rectangle(img_tmp,(0,h-10),(w,h),0,-1)
    img_tmp = cv2.rectangle(img_tmp,(w-10,0),(w,h),0,-1)
    img_tmp = cv2.rectangle(img_tmp,(0,0),(w,10),0,-1)

    return img_tmp

#apply grabcut for a images to get a foreground-only image
#input:a image,id:=
#output:foreground-only image(gray-scale)
def GrabCutFry(img,id=0,outdir='result_grabcut',x_magnification=1.0,y_magnification=1.0):
    print 'x_magnification is:%lf\ny_magnification is:%lf' % (x_magnification,y_magnification,)
    if not type(x_magnification) == float:
        x_magnification = float(x_magnification)
    if not type(y_magnification) == float:
        y_magnification = float(y_magnification)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    if not type(img) == np.ndarray:
        print 'No image data is taken over.'
        raise ValueError('error')
    img_sm = cv2.resize(img,None,fx=x_magnification,fy=y_magnification)
    h,w = img_sm.shape[:2]

    mask = np.zeros(img_sm.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (1,1,w-1,h-1)

    print ('apply grabcut initialzed by a rectangle for the image')


    start_time = time.time()
    cv2.grabCut(img_sm,mask,rect,bgdModel,fgdModel,50,cv2.GC_INIT_WITH_RECT)
    finish_time = time.time()
    print ('finished in %lf[sec]'%(finish_time-start_time,))


    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_sm = img_sm*mask2[:,:,np.newaxis]

    # newmask is the mask image I manually labelled
    newmask = make_mask(img_sm)

    # whereever it is marked white (sure foreground), change mask=1
    # whereever it is marked black (sure background), change mask=0
    mask[newmask == 0] = 0
    mask[newmask == 255] = 1
    mask[newmask == 2] = 2
    mask[newmask == 3] = 3

    print ('apply grabcut initialzed by a mask for the image')
    start_time = time.time()
    mask, bgdModel, fgdModel = cv2.grabCut(img_sm,mask,None,bgdModel,fgdModel,90,cv2.GC_INIT_WITH_MASK)
    finish_time = time.time()
    print ('finished in %lf[sec]'%(finish_time-start_time,))

    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_sm = img_sm*mask[:,:,np.newaxis]
    filename = 'grabcut_result_id=%d.png' % (id)

    img_result = cv2.resize(img_sm,None,fx=1/x_magnification,fy=1/y_magnification)

    img_result = cv2.resize(img_result,None,fx=0.5,fy=0.5)
    img_result = cv2.resize(img_result,None,fx=2,fy=2)

    cv2.imwrite(os.path.join(outdir,filename),img_result)
    print '\nthe image has been saved as "%s"\n'%(os.path.join(outdir,filename),)
    return img_result

#apply convex-hull for a foreground-only image and get 3 sets of gripping points
def convex_hull_fry(img,id=0,outdir = 'result_convexhull'):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    start_time = time.time()
    print 'start convex-hull'
    imhsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    _,_,imgray = cv2.split(imhsv)

    ret,thresh = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #BGR2GRAY

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    result_morphing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours,hierarchy= cv2.findContours(thresh,1,2)

    length = 0
    index = 0
    for i,cnt in enumerate(contours):
        if length < len(cnt):
            length = len(cnt)
            index = i
    cnt = contours[index]
    hull = cv2.convexHull(cnt)

    vectors = []
    for i in range(len(hull)):
        d = range(len(hull))
        d.remove(i)
        for j in d:
            va = hull[i][0]
            vb = hull[j][0]
            v1 = np.array(va) - np.array(vb)
            dots = range(len(hull))
            dots.remove(i)
            dots.remove(j)
            plus=0
            minus=0
            zeros=0
            for k in dots:
                tmp = hull[k][0]
                v2 = va - np.array(tmp)

                if np.cross(v1,v2) > 0:
                    plus += 1
                elif np.cross(v1,v2) < 0:
                    minus += 1
                else:
                    zeros += 0
            if (plus == 0 and minus != 0) or (plus !=0 and minus == 0):
                vectors.append([va,vb])
            else:
                continue
    tmp=0
    maxva=None
    maxvb=None
    length = 0

    for v in vectors:
        va = v[0]
        vb = v[1]

        length = np.linalg.norm((np.array(va)-np.array(vb)),ord=2)
        if length > tmp:
            tmp = length
            maxva = va
            maxvb = vb

    u =  np.array(np.array(maxva)-np.array(maxvb))
    tmp = 0
    L=0
    maxv = 0
    for h in hull:
        v = np.array(np.array(maxva)-np.array(h[0]))
        tmp = abs(np.cross(u,v)/np.linalg.norm(u))
        if tmp > L:
            L = tmp
            maxv = h[0]
    #maxva,maxvb -> 2 sets of points of fry's cavity
    #maxv -> the point of opposite side of cavity
    points = [maxva[1],maxva[0],maxvb[1],maxvb[0],maxv[1],maxv[0]]
    end_time = time.time()
    print 'convex-hull has done in %lf[sec].'%(end_time-start_time,)
    img_draw = img.copy()
    cv2.circle(img_draw,(maxva[0],maxva[1]),3,color=(255,0,0),thickness=-1)
    cv2.circle(img_draw,(maxvb[0],maxvb[1]),3,color=(0,255,0),thickness=-1)
    cv2.circle(img_draw,(maxv[0],maxv[1]),3,color=(0,0,255),thickness=-1)

    filename = 'convexhull_result_id=%d.png' % (id,)
    cv2.imwrite(os.path.join(outdir,filename),img_draw)
    print '\nthe image has been saved as "%s"\n'%(os.path.join(outdir,filename),)
    return points

# Chainer -> OpenCV
def trans_img_cv2(img):
    buf = np.asanyarray(img, dtype=np.uint8).transpose(1, 2, 0)
    dst = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    return dst

# OpenCV -> Chainer
def trans_img_chainer(img):
    buf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = np.asanyarray(buf, dtype=np.float32).transpose(2, 0, 1)
    return dst

#return_coord_for_tirm == True,this function returns rotated img and coord_for_tirm
#return_coord_for_tirm == True,this function returns rotated img and just rotated inputted coords
def rotate_points(img,coords,points,angle):
    import itertools
    h,w = img.shape[:2]
    angle_rad = np.deg2rad(angle)
    img_rot = trans_img_cv2(chainercv.transforms.rotate(trans_img_chainer(img),angle))
    img_trimmed = trim_img(img,coords)

    if len(coords) % 2 == 1:
        print 'the sum of coordinations is odd!'
        raise Valuerror('error')

    corner_offset_x = []
    corner_offset_y = []
    co = np.cos(angle_rad)
    si = np.sin(angle_rad)
    for corner in itertools.product([0,h-1],
                                    [0,w-1]):
        ox = corner[1] * co + corner[0] * si
        oy = -corner[1] * si + corner[0] * co
        corner_offset_x.append(ox)
        corner_offset_y.append(oy)
    xl = []
    yl = []
    for corner_x,corner_y in itertools.product([coords[1],coords[3]],
                                               [coords[0],coords[2]]):
        offset_x =  corner_x * co + corner_y * si - min(corner_offset_x)
        offset_y = -corner_x * si + corner_y * co - min(corner_offset_y)
        xl.append(offset_x)
        yl.append(offset_y)


    min_x = int(min(xl))
    min_y = int(min(yl))
    max_x = int(max(xl))
    max_y = int(max(yl))
    coord_rot = (int(min_y),int(min_x),int(max_y),int(max_x))
    img_rot_trimmed = trim_img(img_rot,coord_rot)

    points_rot = []
    test = []

    corner_offset_x = []
    corner_offset_y = []
    ht,wt=img_trimmed.shape[:2]
    for corner in itertools.product([0,ht-1],
                                    [0,wt-1]):
        ox = corner[1] * co + corner[0] * si
        oy = -corner[1] * si + corner[0] * co
        corner_offset_x.append(ox)
        corner_offset_y.append(oy)

    for y,x in np.array(points).reshape(3,2) :
        rotated_x = x * co + y * si - min(corner_offset_x)
        rotated_y = -x * si + y * co - min(corner_offset_y)
        points_rot.append([int(rotated_y), int(rotated_x)])
    points_rot = np.array(points_rot).ravel()

    return img_rot_trimmed,coord_rot,np.array(points_rot).ravel()

def slide_points(img,coords,points,x_stride,y_stride):
    h,w = img.shape[:2]
    ymin,xmin,ymax,xmax = coords
    point1_y,point1_x,point2_y,point2_x,point3_y,point3_x = points
    if (x_stride == 0) and (y_stride == 0):
        print 'both x and y stride are zero!'
        return None

    if x_stride >= 0:
        if y_stride >= 0:
            new_ymin = ymin+abs(y_stride)
            new_xmin = xmin+abs(x_stride)
            new_ymax = ymax+abs(y_stride)
            new_xmax = xmax+abs(x_stride)

        elif y_stride < 0:
            new_ymin = ymin-abs(y_stride)
            new_xmin = xmin+abs(x_stride)
            new_ymax = ymax-abs(y_stride)
            new_xmax = xmax+abs(x_stride)

    elif x_stride <= 0:
        if y_stride >= 0:
            new_ymin = ymin+abs(y_stride)
            new_xmin = xmin-abs(x_stride)
            new_ymax = ymax+abs(y_stride)
            new_xmax = xmax-abs(x_stride)

        elif y_stride < 0:
            new_ymin = ymin-abs(y_stride)
            new_xmin = xmin-abs(x_stride)
            new_ymax = ymax-abs(y_stride)
            new_xmax = xmax-abs(x_stride)

    if not ((new_ymin <= point1_y+ymin <= new_ymax)and(new_xmin <= point1_x+xmin <= new_xmax)):
        return None
    if not ((new_ymin <= point2_y+ymin <= new_ymax)and(new_xmin <= point2_x+xmin <= new_xmax)):
        return None
    if not ((new_ymin <= point3_y+ymin <= new_ymax)and(new_xmin <= point3_x+xmin <= new_xmax)):
        return None

    return [new_ymin,new_xmin,new_ymax,new_xmax]

#function for size fitting
#compare width and height of image to the other ones
def check_maximum_width_and_height(wh_info):
    w = []
    h = []
    for entry in wh_info:
        w.append(entry[0])
        h.append(entry[1])

    return max(w),max(h)

def make_dataset_for_VIA(json_filename,base_dir,has_size_fitting=False):
    import json
    import os
    import sys
    sys.stdout.flush()
    print 'the paramater of "has_size_fitting" is %s' % (str(has_size_fitting),)

    with open(os.path.join(base_dir,json_filename), 'r') as fp:
        tree = json.load(fp)
    result = []
    for v in tree['_via_img_metadata'].itervalues():
        filename = v['filename']
        full_filename = os.path.join(base_dir, filename)
        if not os.path.isfile(full_filename):
            print('there is no file named %s!'%(full_filename,))
            continue
            # raise IOError(full_filename)
        sys.stdout.write('.')
        sys.stdout.flush()
        basename, ext = os.path.splitext(filename)
        regions = v['regions']
        arr = cv2.imread(os.path.join(base_dir, filename))
        bboxes = []
        wh = []
        img = []

        for r in v['regions']:
            attr = r['shape_attributes']
            width = attr['width']
            height = attr['height']

            xmin = attr['x']
            ymin = attr['y']
            xmax = xmin + width
            ymax = ymin + height
            bboxes.append([ymin, xmin, ymax, xmax])
            wh.append([width,height])
            if not has_size_fitting:
                img.append(trim_img(arr,[ymin, xmin, ymax, xmax]))
        if len(bboxes) == 0:
            continue
        if has_size_fitting:
            stride = 5
            maxw,maxh = check_maximum_width_and_height(wh)
            for index,(ymin,xmin,ymax,xmax) in enumerate(bboxes):
                width_tmp = xmin-xmax
                xmin -= stride
                xmax = xmin + maxw+stride
                bboxes[index][1] = xmin
                bboxes[index][3] = xmax
                height_tmp = xmin-xmax
                ymax = ymin + maxh+stride
                ymin -= stride
                bboxes[index][2] = ymax
                bboxes[index][0] = ymin

            for coords in bboxes:
                img.append(trim_img(arr,coords))

        bboxes = np.array(bboxes, np.float32)
        result.append((filename,bboxes,wh,img))
    sys.stdout.write('\n')
    sys.stdout.flush()
    return result

def make_json_for_points(VIA_filename,base_dir,size_fitting=False,savename='train_info.json',save_result=True,takeoverid=0):
    tree = make_dataset_for_VIA(VIA_filename,base_dir,has_size_fitting=size_fitting)
    result = []
    ID = 0
    for filename,coords,wh,images in tree:
        lenght_of_result = len(result)
        for I,(img,coord) in enumerate(zip(images,coords)):
            sys.stdout.write('.')
            sys.stdout.flush()
            ID = I+lenght_of_result+takeoverid
            dic = {}
            dic['ID'] = str(ID)
            dic['filename'] = str(filename)
            dic['coords'] = [str(c) for c in coord]
            now = time.time()
            points = convex_hull_fry(GrabCutFry(img,id=ID,x_magnification=4.,y_magnification=4.),id=ID)
            print(time.time()-now)
            dic['points'] = [str(p) for p in points]
            result.append(dic)

    if save_result:
        json_dumper(result,output_name=savename)
    return result

def load_json_for_points(json_filename,json_basedir='.',skip_fielname=None,basedir='./data',checker=False):
    print 'start loading info from json file.'
    result = []
    inds = 0
    filename = os.path.join(json_basedir,json_filename)
    with open(json_filename,'r') as fp:
        tree = json.load(fp)
    if not skip_fielname is None:
        with open(skip_fielname,'r') as fp:
            skip = fp.read().split()
        skip = list(set(skip))
    else:
        skip = []
    for entry in tree:
        if entry['ID'] in skip:
            if checker:
                print 'skipped! %s' %(entry['ID'],)
            continue
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            ID = int(entry['ID'])
            filename = entry['filename']
            coords = [int(float(i)) for i in entry['coords']]
            points =  [int(float(i)) for i in entry['points']]
            img_raw = cv2.imread(os.path.join(basedir,filename))
            img = trim_img(img_raw,coords)
            # img_output = draw_marker(img,points)
            # cv2.imwrite('./wholedata/%s_%s.png'%(ID,str(points)),img_output)
            # print ID
            inds +=1
            result.append([ID,filename,coords,points,img])
    print '\nloading has done.'
    return result

#encode a dictionaty and save it as json style file
def json_dumper(dic,output_name='default.json',relative_output_dir ='.',encord = False):
    import json
    import os
    global output_dir_prefix
    output_dir = os.path.join(output_dir_prefix,relative_output_dir)
    if not os.path.isdir(output_dir):
        os.makedir(output_dir)
    if encord:
        result = json.dumps(dic)
    else:
        result = dic
    with open(os.path.join(output_dir,output_name),'w') as fp:
        json.dump(result,fp,indent=1,sort_keys=False,ensure_ascii=False)
    print 'file has been saved as %s' %(os.path.join(output_dir,output_name),)

def trim_img(img,coords):
    #img[height,width]
    xmin = int(min(coords[0],coords[2]))
    xmax = int(max(coords[0],coords[2]))
    ymin = int(min(coords[1],coords[3]))
    ymax = int(max(coords[1],coords[3]))

    return img[xmin:xmax,ymin:ymax]

def data_augmentation(datalist,degree_gap=30,shift_range_max = 10,stride=2,basedir='./data'):
    print '\nstart data augmentation.'
    result = []
    length_of_datalist = len(datalist)
    last_filename = None
    for i_1,(ID,filename,coords,points,img) in enumerate(datalist):
        sys.stdout.write('.')
        sys.stdout.flush()
        if not filename == last_filename:
            o_img = cv2.imread(os.path.join(basedir,filename))
        length_of_result = len(result)
        for i_2,(angle) in enumerate(range(degree_gap,360,degree_gap)):
        #rotated_coords,rotated_image
            r_img,r_coords,r_points = rotate_points(o_img,coords,points,angle)
            r_points = [int(p) for p in r_points]
            r_id = '%d-%d'%(i_1,i_2,)
            result.append([r_id,filename,r_coords,r_points,r_img])


        for i_3,range_shift in enumerate(range(1,shift_range_max,stride)):
            for shift_times in range(8):
                length_of_result = len(result)
                shift03 = base_10_to_n(shift_times,3)
                if not len(shift03) == 2 :
                    shift03 = '0' + shift03

                if shift03[1] == '0':
                    x_stride = -range_shift
                elif shift03[1] == '1':
                    x_stride = range_shift
                elif shift03[1] == '2':
                    x_stride = 0

                if shift03[0] == '0':
                    y_stride = -range_shift
                elif shift03[0] == '1':
                    y_stride = range_shift
                elif shift03[0] == '2':
                    y_stride = 0
                s_id = '%d-%d'%(i_1,i_2+i_3+1,)
                s_coords = slide_points(o_img,coords,points,x_stride,y_stride)
                if not (s_coords is None):
                    s_img = trim_img(o_img,s_coords)
                    result.append([s_id,filename,s_coords,points,s_img])
        last_filename = filename

    return result

def base_10_to_n(X, n):
    if (int(X/n)):
        return base_10_to_n(int(X/n), n)+str(X%n)
    return str(X%n)

# def data_augmentation_with_chainercv(datalist,degree_gap=30,shift_range_max = 10,stride=2,basedir='./data',cheker=True):
#     print '\nstart data augmentation using chainercv.'
#     result = []
#     length_of_datalist = len(datalist)
#     last_filename = None
#     for ID1,filename,coords,points,img in datalist:
#         sys.stdout.write('.')
#         sys.stdout.flush()
#         if not filename == last_filename:
#             original_img = chainercv.utils.read_image(os.path.join(basedir,filename))
#         length_of_result = len(result)
#         h,w = img.shape[:2]
#         img_chainer = trans_img_chainer(img)
#         for ID2,(angle) in enumerate(range(degree_gap,360,degree_gap)):
#         #rotated_coords,rotated_image
#             sys.stdout.write(',')
#             sys.stdout.flush()
#             original_H,original_W,_ = original_img.shape
#             rotated_image = chainercv.transforms.rotate(original_img,angle)
#             rotated_coords = chainercv.transforms.rotate_bbox(coords,angle,(original_H,original_W))
#             rotate_img_cropped = chainercv.transforms.rotate(img_chainer,angle)
#
#             angle_rad = np.deg2rad(angle)
#             corner_offset_x = []
#             corner_offset_y = []
#             co = np.cos(np.deg2rad(angle))
#             si = np.sin(np.deg2rad(angle))
#             for corner in itertools.product([0,h-1],
#                                             [0,w-1]):
#                 ox = corner[1] * co + corner[0] * si
#                 oy = -corner[1] * si + corner[0] * co
#                 corner_offset_x.append(ox)
#                 corner_offset_y.append(oy)
#             for y,x in np.array(points).reshape(3,2) :
#                 rotated_x = x * co + y * si - min(corner_offset_x)
#                 rotated_y = -x * si + y * co - min(corner_offset_y)
#                 rotated_points.append([int(rotated_y-0.5), int(rotated_x-0.5)])
#             rotated_points = np.array(points_rot).ravel()
#             rotated_ID = '%d-%d'%(ID1,ID2,)
#             rotated_img_trim = trim_img(trans_img_cv2(rotated_image),coords)
#
#             result.append([rotated_ID,filename,rotated_coords,rotated_points,rotated_img_trim])
#             print 'hoge'
#             raise
#         for ID3,range_shift in enumerate(range(1,shift_range_max,stride)):
#             sys.stdout.write('*')
#             sys.stdout.flush()
#             for shift_times in range(8):
#                 length_of_result = len(result)
#                 shift03 = base_10_to_n(shift_times,3)
#                 if not len(shift03) == 2 :
#                     shift03 = '0' + shift03
#
#                 if shift03[1] == '0':
#                     x_stride = -range_shift
#                 elif shift03[1] == '1':
#                     x_stride = range_shift
#                 elif shift03[1] == '2':
#                     x_stride = 0
#
#                 if shift03[0] == '0':
#                     y_stride = -range_shift
#                 elif shift03[0] == '1':
#                     y_stride = range_shift
#                 elif shift03[0] == '2':
#                     y_stride = 0
#                 s_id = '%d-%d'%(ID1,ID2+ID3+1,)
#                 s_coords = slide_points(original_img,coords,points,x_stride,y_stride)
#                 if not (s_coords is None):
#                     s_img = trim_img(original_img,s_coords)
#                     result.append([s_id,filename,s_coords,points,s_img])
#         last_filename = filename
#
#     return result

def main():
    dataset_a = make_json_for_points('fried_shrimp.json','data',size_fitting=False,save_result=False)

if __name__ == '__main__':
    dataset_a = make_json_for_points('fried_shrimp.json','data',size_fitting=True,savename='train_info_1.json')
