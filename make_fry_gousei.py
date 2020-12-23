# coding:utf-8
import cv2
from utils_for_point_estimation import *
import re
import argparse

import argparse
import numpy as np
import sys
import os
from timeit import default_timer as timer

import chainer
from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger
import chainer.functions as F
import chainer.links as L

import chainercv
from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links import FasterRCNNVGG16
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain
from chainercv import transforms

from chainer_npz_with_structure import make_serializable_object
from chainer_npz_with_structure import save_npz_with_structure
from chainer_npz_with_structure import load_npz_with_structure

import cupy
class ConcatenatedDataset(chainer.dataset.DatasetMixin):#chainerで独自データセットクラス(ConcatenatedDataset)を作っている。chainer.dataset.DatasetMixinを引数にすれば作れる。
    """
独自データセットを作る時のルール
    1.データセットにするクラスは chainer.dataset.DatasetMixinを継承する
    2.内部に持っているデータの数を返却する __len__(self) メソッドを実装する。このメソッドは整数でデータ数を返却する
    3.i番目のデータを取得する get_example(self, i) メソッドを実装する。このメソッドは、
        画像配列などのデータ
        ラベル
の2つを返却する（return image_array, label みたいな感じで）
    """
    def __init__(self, *datasets):
        self._datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def get_example(self, i):
        #print ('(i:%d)' % (i,))
        #sys.stdout.flush()
        if i < 0:
            raise IndexError
        for dataset in self._datasets:
            if i < len(dataset):
                return dataset[i]
            i -= len(dataset)
        raise IndexError

class Transform(object):

    def __init__(self, faster_rcnn):#インスタンス化する時に __init__ が、自動的に呼び出される
        self.faster_rcnn = faster_rcnn

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip 水平に反転.  transforms.random_flip:chainerCVに実装されている関数。画像を反転してくれる
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)

	#↓transforms.flip_bbox:chainerCVにある関数。画像のトリミングされた領域に収まるようにバウンディングボックスを変換します。この方法は主に画像のトリミングと一緒に使用されます。このメソッドは、translate_bbox（）のようにバウンディングボックスの座標を変換します。さらに、この関数は、トリミングされた領域内に収まるようにバウンディングボックスを切り詰めます。バウンディングボックスがトリミング領域と重ならない場合、このバウンディングボックスは削除されます。
        bbox = transforms.flip_bbox( #バウンディングボックス（Bounding Box）とは，図形をちょうど囲うのに必要な大きさの，四角い箱 （矩形）のこと
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale

global_trainer = None
def my_save_model(trainer): #作ったモデルを保存する関数
    global global_trainer
    global_trainer = trainer
    model = trainer.updater.get_optimizer('main').target.faster_rcnn
    filename_format = '{.out}/model_ep{.updater.epoch}.npz'
    model_filename = filename_format.format(trainer,trainer)
    print("Save the current model to %s ..." % model_filename)
    time_save_start = timer()
    save_npz_with_structure(model_filename, model)
    time_save_end = timer()
    print(" finished in (%e[sec])." % (time_save_end - time_save_start))
    sys.stdout.flush()


# def calc_cw(dataset,label_names,gpu=0):
#     from collections import Counter
#     import numpy as np
#
#
#     temp = []
#     result = np.zeros(len(label_names),dtype=int)
#     cw = []
#     for r in dataset:
#         for l in r[2]:
#             temp.append(l)
#
#     for k,v in Counter(temp).items():
#         result[k] = v
#
#     cw.append(1)
#     for i in result:
#         if i != 0:
#             cls_wit = sum(result)/i
#             if cls_wit >= 10:
#                 cw.append(10)
#             else:
#                 cw.append(sum(result)/i)
#         else:
#             cw.append(1)
#     return cupy.core.core.array(cw).astype(np.float32)

# def make_dataset_from_json2(json_file,base_dir,label_names,default_label_name=None,valid_filenames=None):
#     import json
#     import os
#     import sys
#
#     sys.stdout.flush()
#     with open(base_dir+json_file, 'r') as fp:
#         tree = json.load(fp)
#     result = []
#     for v in tree['_via_img_metadata'].itervalues():
#         filename = v['filename']
#         full_filename = os.path.join(base_dir, filename)
#         if valid_filenames and not filename in valid_filenames:
#             continue
#         elif not os.path.isfile(full_filename):
#             continue
#         sys.stdout.write('.')
#         sys.stdout.flush()
#         basename, ext = os.path.splitext(filename)
#         regions = v['regions']
#         arr = chainercv.utils.read_image(os.path.join(base_dir, filename))
#         bboxes = []
#         labels = []
#         for r in v['regions']:
#             attr = r['shape_attributes']
#             xmin = attr['x']
#             ymin = attr['y']
#             xmax = xmin + attr['width']
#             ymax = ymin + attr['height']
#             if attr['width']>attr['height']:
#                 w.append(attr['width'])
#                 h.append(attr['height'])
#             else:
#                 h.append(attr['width'])
#                 w.append(attr['height'])
#
#             label = None
#             if r['region_attributes'] == {}:
#                 print ('\n')
#                 print('There is a region without attributes!: r=(%s)' % (r,))
#                 print('filename="%s"' % (filename,))
#                 continue
#             else:
#                 bboxes.append([ymin, xmin, ymax, xmax])
#                 if 'class' in r['region_attributes']:
#                     label_name = r['region_attributes']['class']
#                 else:
#                     print("r: %s" % (r, ))
#                     print("r['region_attributes]: %s" % (r['region_attributes'], ))
#                     print("filename: %s" % (filename, ))
#                     label_name = default_label_name
#                 label = label_names.index(label_name)
#                 labels.append(label)
#         if len(bboxes) == 0:
#             continue
#         bboxes = np.array(bboxes, np.float32)
#         labels = np.array(labels, np.int32)
#         result.append((arr, bboxes, labels, filename))
#     sys.stdout.write('\n')
#     sys.stdout.flush()
#     print 'ave_width:'
#     print sum(w)/len(w)
#     print 'ave_height:'
#     print sum(h)/len(h)
#     print 'max_width'
#     print max(w)
#     print 'max_height:'
#     print max(h)
#
#     return result

def getMask(frame, lower_color=(1,1,1), upper_color=(255,255,255)):#Mask-r-cnn用の関数
  # HSVによる画像情報に変換
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # ガウシアンぼかしを適用して、認識精度を上げる
  blur = cv2.GaussianBlur(hsv, (9, 9), 0)

  # 指定した色範囲のみを抽出する
  color = cv2.inRange(blur, lower_color, upper_color)
  # inRange(画像,(h,s,vのそれぞれ下限),(上限))


  # オープニング・クロージングによるノイズ除去
  element8 = np.ones((5,5),np.uint8)
  oc = cv2.morphologyEx(color, cv2.MORPH_OPEN, element8,iteration=10)
  oc = cv2.morphologyEx(oc, cv2.MORPH_CLOSE, element8,iteration=10)
  oc[np.where((oc == 255))] = 1

  return oc

def make_canvas(size=(1200,1200,3),color=(168,115,28)): #キャンバスを作成(絵を書く場所作るみたいな感じ) ※綴りが本当はcanvas
    canvas = np.zeros(size,np.uint8) #要素の値が全てゼロの配列を作成するにはzeros()関数を使用します。np.zeros(形状)という書き方をするnp.uint8というデータ型で今回は指定している。
    canvas[:] = color
    return canvas

def make_bbox(img_original,lower_color=(1,1,1), upper_color=(255,255,255)):
    img = img_original.copy()#img_originalをコピー(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#コピーしたimgをBGrからｈｓｖに変換(hsv)
    blur = cv2.GaussianBlur(hsv, (9, 9), 0)#hsvにガウシアンフィルタをかける9x9のフィルタ.　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
    color = cv2.inRange(blur, lower_color, upper_color)#???抽出する色の上限下限を作成してinRange関数でマスクを取得します。そのマスクを元画像に適応するとこで、範囲内の色以外を黒くしている。???ここ微妙
    element8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    oc = cv2.morphologyEx(color, cv2.MORPH_OPEN, element8)#オープニング処理. 収縮の後に膨張 をする処理です．ノイズ除去に有効で、関数は cv2.morphologyEx() を使います．
    oc = cv2.morphologyEx(oc, cv2.MORPH_CLOSE, element8)#クロージング処理はオープニング処理の逆の処理を指し， 膨張の後に収縮 をする処理です．前景領域中の小さな(黒い)穴を埋めるのに役立ちます．オープニングと同様 cv2.morphologyEx() 関数を使いますが，第2引数のフラグに cv2.MORPH_CLOSE を指定する点が違います．
    contours, hierarchy = cv2.findContours(oc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#2値画像から輪郭を抽出します．この輪郭は，形状の解析や物体検出，物体認識を行うための有効な手段.
    #輪郭抽出のモード：CV_RETR_EXTERNAL 最も外側の輪郭のみを抽出します.  輪郭の近似手法：CV_CHAIN_APPROX_SIMPLE 水平・垂直・斜めの線分を圧縮し，それらの端点のみを残します．例えば，まっすぐな矩形の輪郭線は，4つの点にエンコードされます．contours:抽出された輪郭。　hierarchy:オプション．画像のトポロジーに関する情報を含む出力ベクトル．これは，輪郭数と同じ数の要素を持ちます．△
    if len(contours) > 0:
        contours.sort(key=cv2.contourArea, reverse=True)
        cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)#boundingRect.外接矩形を取得。四角の長方形でエビフライを囲む感じ。
    return (x,y,w,h)

def make_fry_data(dir,color=(0,0,0),label='default',resize=None):#dir=grabcutが入ってるフォルダ名。「good」,「bad」のこと
    import glob
    full_dir = './%s/*.png'%(dir,)
    filenames = glob.glob(full_dir,)
    result=[]
    index = 0
    # w=[]
    # h=[]
    print 'loading images.'
    for filename in filenames:
        sys.stdout.write('.')
        sys.stdout.flush()
        img = cv2.imread(filename)#cv2.imread()で画像ファイルから読み込み.filenames(good or badのgrabcut画像のまとまり)をfilenameとして一枚ずつimgへ入れる
        img[np.where((img == [0,0,0]).all(axis=2))] = color#黒の領域に今回使う背景の色を当てはめる
        if not resize is None:#引数がNoneじゃない場合、つまりresizeに何か値が与えられた時はcv2.resizeでサイズを変更するということ。
            img = cv2.resize(img,dsize=None,fx=resize,fy=resize)
        #bboxes = make_bbox(img)
        w,h,_ = img.shape
        bboxes = (0,0,h,w)

        result.append((img,bboxes,label,filename,resize))
    # print 'ave_width:'
    # print sum(w)/len(w)
    # print 'ave_height:'
    # print sum(h)/len(h)
    # print 'max_width'
    # print max(w)
    # print 'max_height:'
    # print max(h)
    sys.stdout.write('\n')
    sys.stdout.flush()
    print 'load has been done.'
    return result#img,bbox,label情報の配列resultを返す。

#この関数のコードのままだと訓練画像・テスト画像(合成)が混ざってしまうという問題がある+重なった合成画像が作成できない → make_composite_img2として書きなおした。
def make_composite_img(csize=(3600,3600,3),resizef=0.7,color=(168,115,28),good_dir='./good',bad_dir='./bad',times=200,Train_flag=True,rotate_flag=True):#resizef=0.7にするとgoodフライのサイズがbadフライに対してちょうどよくなる.csize3600 3=RGB

    result = []
    result_train=[]
    result_test=[]
    #ここのgood_dirの時点で、good_dirをgood_train_grabcutとgood_test_grabcutの二個にしておくといい。
    good_fry_data_original = make_fry_data(good_dir,label=0,resize=resizef)#[画像,句形情報,ラベル名,ファイル名、リサイズ]==[img,bboxes,label,filename,resize]の５つで１セットのデータがたくさんあるリストが作成される。
    #good_fry_data_originalの中身のイメージ:[[img1, bboxes1, 0, '0.png', 0.7], [img2, bboxes2, 0, '1.png', 0.7], [img3, bboxes3, 0, '2.png', 0.7], .......]という感じ
    #label0:good, label 1:bad
    good_fry_data_original = np.random.permutation(good_fry_data_original)#shuffle,順番変わっただけ

    bad_fry_data_original = make_fry_data(bad_dir,label=1)
    #bad_fry_data_originalの中身のイメージ:[[img1, bboxes1, 1, '0.png', 0.7], [img2, bboxes2, 1, '1.png', 0.7], [img3, bboxes3, 1, '2.png', 0.7], .......]という感じ ※labelがbadなので'1'になっている。
    bad_fry_data_original = np.random.permutation(bad_fry_data_original)

    good_fry_data_train,good_fry_data_test = np.split(good_fry_data_original,[int(np.floor(0.8*len(good_fry_data_original)))])
    bad_fry_data_train,bad_fry_data_test = np.split(bad_fry_data_original,[int(np.floor(0.8*len(good_fry_data_original)))])

    good_fry_data = good_fry_data_train
    bad_fry_data = bad_fry_data_train

    for t in range(times):
        sys.stdout.write('.')
        sys.stdout.flush()
        canvas = make_canvas(csize,color)
        H_o,W_o,_ = canvas.shape
        #these 2 should be devided for test#


        max = 16
        # num = np.random.randint(0,max+1)

        num = 8
        #-*- np.random.randint(始点,終点+1,個数) -*-#
        good_num = np.random.randint(0,len(good_fry_data),num)
        bad_num = np.random.randint(0,len(bad_fry_data),max-num)

        datas = []
        # print 'good_num:%d'%(num,)
        # print 'bad_num:%d'%(20-num,)
        for i in good_num:#選び出してきた番号のリスト
            datas.append(good_fry_data[i])
        for i in bad_num:
            datas.append(bad_fry_data[i])
        datas = np.random.permutation(datas)#良品不良品8個ずつ(良品num個,不良品max-num)シャッフルする

        ypoints = [40,840,1640,2440]
        xpoints = [40,840,1640,2440]
        bboxes = []
        labels = []
        image_placement_info = []


        for i,y in enumerate(ypoints):
            for j,x in enumerate(xpoints):#i,j ninannbanmekahaitteru
                x_offset = np.random.randint(-30, 31) + x#offset=ズレ
                y_offset = np.random.randint(-30, 31) + y
                ind = i*len(xpoints) + j
                img = datas[ind][0]
                # xb,yb,wb,hb = datas[ind][1]
                label = datas[ind][2]
                filename = datas[ind][3]
                resize = datas[ind][4]

                img_rotated,_,angle = rotate_img(img,[0,340,20],cv2_flag=True)
                img_tocanvas = img_rotated.copy()
                img_tocanvas[np.where((img_tocanvas == [0,0,0]).all(axis=2))] = color#四隅のあまりを埋めてる
                bbox_info = make_bbox(img_rotated)#回転したデータの中で、どこにフライが写ってるか、フライを囲む矩形情報を計算。
                H,W,_ = img_rotated.shape

                miny = y_offset#zure wo sakusei
                maxy = y_offset+H#開店後の画像の高さH
                minx = x_offset
                maxx = x_offset+W
                # print canvas.shape
                # print img_rotated.shape
                canvas[miny:maxy,minx:maxx] = img_tocanvas

                miny = bbox_info[1] + y_offset#y_offsetはキャンバス上での座標。bbox_info[]回転後の矩形の中でフライがどこに写っているか。
                maxy = bbox_info[1]+bbox_info[3] + y_offset#bbox_info[3]=h
                minx = bbox_info[0] + x_offset
                maxx = bbox_info[0]+bbox_info[2] + x_offset

                bboxes.append([miny,minx,maxy,maxx])#ここでひとつのフライを特定の場所に配置
                labels.append(label)
                #filename,resize,label,x_offset,y_offset,ind,angle,color,bbox_infoを保存
                image_placement_info.append((filename,resize,label,x_offset,y_offset,ind,angle,color,bbox_info))

        resized_canvas = trans_img_chainer(cv2.resize(canvas,dsize=None,fx=0.3,fy=0.3))
        _,H_r,W_r = resized_canvas.shape
        bboxes = np.array(bboxes)

        resized_bboxes = chainercv.transforms.resize_bbox(bboxes, (H_o,W_o), (H_r,W_r))

        info_with_resized = []
        for i in range(len(bboxes)):
          info_with_resized.append(image_placement_info[i]+(resized_bboxes[i],))

        result_train.append([resized_canvas,resized_bboxes,labels,info_with_resized])

    sys.stdout.write('\n')
    sys.stdout.flush()

    good_fry_data = good_fry_data_test
    bad_fry_data = bad_fry_data_test

    for t in range(times):
        sys.stdout.write('.')
        sys.stdout.flush()
        canvas = make_canvas(csize,color)
        H_o,W_o,_ = canvas.shape
        #these 2 should be devided for test#


        max = 16
        # num = np.random.randint(0,max+1)

        num = 8
        #-*- np.random.randint(始点,終点+1,個数) -*-#
        good_num = np.random.randint(0,len(good_fry_data),num)
        bad_num = np.random.randint(0,len(bad_fry_data),max-num)

        datas = []
        # print 'good_num:%d'%(num,)
        # print 'bad_num:%d'%(20-num,)
        for i in good_num:
            datas.append(good_fry_data[i])
        for i in bad_num:
            datas.append(bad_fry_data[i])
        datas = np.random.permutation(datas)

        ypoints = [40,840,1640,2440]
        xpoints = [40,840,1640,2440]
        bboxes = []
        labels = []

        for i,y in enumerate(ypoints):
            for j,x in enumerate(xpoints):
                x_offset = np.random.randint(-30, 31) + x
                y_offset = np.random.randint(-30, 31) + y
                ind = i*len(xpoints) + j
                img = datas[ind][0]
                # xb,yb,wb,hb = datas[ind][1]
                label = datas[ind][2]
                img_rotated,_ = rotate_img(img,[0,340,20],cv2_flag=True)
                img_tocanvas = img_rotated.copy()
                img_tocanvas[np.where((img_tocanvas == [0,0,0]).all(axis=2))] = color
                bbox_info = make_bbox(img_rotated)
                H,W,_ = img_rotated.shape

                miny = y_offset
                maxy = y_offset+H
                minx = x_offset
                maxx = x_offset+W
                # print canvas.shape
                # print img_rotated.shape
                canvas[miny:maxy,minx:maxx] = img_tocanvas

                miny = bbox_info[1] + y_offset
                maxy = bbox_info[1]+bbox_info[3] + y_offset
                minx = bbox_info[0] + x_offset
                maxx = bbox_info[0]+bbox_info[2] + x_offset

                bboxes.append([miny,minx,maxy,maxx])
                labels.append(label)
        resized_canvas = trans_img_chainer(cv2.resize(canvas,dsize=None,fx=0.3,fy=0.3))
        _,H_r,W_r = resized_canvas.shape
        bboxes = np.array(bboxes)

        resized_bboxes = chainercv.transforms.resize_bbox(bboxes, (H_o,W_o), (H_r,W_r))
        result_test.append([resized_canvas,resized_bboxes,labels])
    sys.stdout.write('\n')
    sys.stdout.flush()

    return result_train,result_test


#重なった合成画像を作成するための関数(新しく自分で作成)
def make_composite_img2(csize=(3600,3600,3),resizef=0.7,color=(168,115,28),good_dir='./good',bad_dir='./bad',times=200,Train_flag=True,rotate_flag=True):#resizef=resize file=大きさ調整するために入れてる。画像大きすぎるとメモリ圧迫するから。csize3600 3=RGB
    #make_composite_imgがトレインとテスト同時だったのを分割したプログラム(自分でやった)
    result = []
    #result_train=[]
    #result_test=[]
    #ここのgood_dirの時点で、good_dirをgood_train_grabcutとgood_test_grabcutの二個にしておくといい。


    #自分でディレクトリ分けて使うならこうしたらいいんかな?→いけた
    #good_fry_data= make_fry_data(good_dir, label=0, resize=resizef)←リサイズは前もってやるようにした。
    #[画像,句形情報,ラベル名,ファイル名、リサイズ]==[img,bboxes,label,filename,resize]の５つで１セットのデータがたくさんあるリストが作成される。
    good_fry_data = make_fry_data(good_dir, label=0)
    good_fry_data= np.random.permutation(good_fry_data)
    #good_fry_dataの中身のイメージ:[[img1, bboxes1, 0, '0.png', 0.7], [img2, bboxes2, 0, '1.png', 0.7], [img3, bboxes3, 0, '2.png', 0.7], .......]という感じ。これがpermutation()で順番ランダムにシャッフルされてる。

    bad_fry_data = make_fry_data(bad_dir, label=1)
    bad_fry_data = np.random.permutation(bad_fry_data)
    #bad_fry_dataの中身のイメージ:[[img1, bboxes1, 1, '0.png', 0.7], [img2, bboxes2, 1, '1.png', 0.7], [img3, bboxes3, 1, '2.png', 0.7], .......]という感じ ※labelがbadなので'1'になっている。


    for t in range(times):#timesは合成画像を作る枚数
        sys.stdout.write('.')
        sys.stdout.flush()
        canvas = make_canvas(csize,color)#(3600,3600,)のcolor(168,115,28)の背景を作成
        H_o,W_o,_ = canvas.shape
        #these 2 should be devided for test#


        max = 16
        #max = 20
        # num = np.random.randint(0,max+1)

        num = 8
        #-*- np.random.randint(始点,終点+1,個数) -*-#
        good_num = np.random.randint(0,len(good_fry_data),num)#0~goodfrydataの数(50)まででランダムにnum(8個)取り出す。ex.[2,18,29,35,9,6,0,49]
        bad_num = np.random.randint(0,len(bad_fry_data),max-num)

        datas = []
        # print 'good_num:%d'%(num,)
        # print 'bad_num:%d'%(20-num,)
        for i in good_num:#選び出してきた番号のリスト
            datas.append(good_fry_data[i])
        for i in bad_num:
            datas.append(bad_fry_data[i])
        datas = np.random.permutation(datas)#良品不良品8個ずつ(良品num個,不良品max-num)ランダムにシャッフルする

        #ypoints = [40,840,1640,2440]
        #xpoints = [40,840,1640,2440]#変更前(初期の位置)

        ypoints = [240,940,1840,2540]#フライの始点(変更後)、もう少し中心にフライが寄るように調整した。
        xpoints = [240,940,1840,2540]

        bboxes = []
        labels = []
        image_placement_info = []


        for i,y in enumerate(ypoints):
            for j,x in enumerate(xpoints):#i,j ninannbanmekahaitteru
                #x_offset = np.random.randint(-30, 31) + x#offset=ズレをつくっている
                x_offset = np.random.randint(-240, 241) + x#フライ間の重なりを作るためにoffsetを大きくした。
                y_offset = np.random.randint(-240, 241) + y
                ind = i*len(xpoints) + j
                img = datas[ind][0]#中身のイメージ[img5, bboxes5, 0, '5.png', 0.7],[img1, bboxes1, 1, '0.png', 0.7]
                # xb,yb,wb,hb = datas[ind][1]
                label = datas[ind][2]
                filename = datas[ind][3]
                resize = datas[ind][4]

                img_rotated,_,angle = rotate_img(img,[0,340,20],cv2_flag=True)
                img_tocanvas = img_rotated.copy()
                #img_tocanvas[np.where((img_tocanvas == [0,0,0]).all(axis=2))] = color#四隅のあまりを埋めてる
                bbox_info = make_bbox(img_rotated)#回転したデータの中で、どこにフライが写ってるか、フライを囲む矩形情報を計算。
                H,W,_ = img_rotated.shape

                miny = y_offset#zure wo sakusei
                maxy = y_offset+H#開店後の画像の高さH
                minx = x_offset
                maxx = x_offset+W
                # print canvas.shape
                # print img_rotated.shape

                #重なっているフライ画像にも対応するために新規にコードを追加(mask)
                target_region=canvas[miny: miny + H, minx: minx + W]
                canvas[miny: miny + H, minx: minx+W] = np.where(img_tocanvas > 0, img_tocanvas, target_region)
                #canvas[minx:maxy,minx:maxx] = img_tocanvas


                miny = bbox_info[1] + y_offset#y_offsetはキャンバス上での座標。bbox_info[]回転後の矩形の中でフライがどこに写っているか。
                maxy = bbox_info[1]+bbox_info[3] + y_offset#bbox_info[3]=h
                minx = bbox_info[0] + x_offset
                maxx = bbox_info[0]+bbox_info[2] + x_offset



                bboxes.append([miny,minx,maxy,maxx])#ここでひとつのフライを特定の場所に配置
                labels.append(label)
                #filename,resize,label,x_offset,y_offset,ind,angle,color,bbox_infoを保存
                image_placement_info.append((filename,resize,label,x_offset,y_offset,ind,angle,color,bbox_info))

        resized_canvas = trans_img_chainer(cv2.resize(canvas,dsize=None,fx=0.3,fy=0.3))
        _,H_r,W_r = resized_canvas.shape
        bboxes = np.array(bboxes)

        resized_bboxes = chainercv.transforms.resize_bbox(bboxes, (H_o,W_o), (H_r,W_r))
        info_with_resized = []
        for i in range(len(bboxes)):
          info_with_resized.append(image_placement_info[i]+(resized_bboxes[i],))

        result.append([resized_canvas,resized_bboxes,labels,info_with_resized])

    sys.stdout.write('\n')
    sys.stdout.flush()

    return result


# def make_composite_img_for_mask(csize=(3600,3600,3),resizef=0.7,color=(168,115,28),good_dir='./good',bad_dir='./bad',times=200,Train_flag=True,rotate_flag=True):#今回は使ってない。mask-r-cnnを使うとき用(途中で断念してる)
#
#     result = []
#     good_fry_data_original = make_fry_data(good_dir,label=0,resize=resizef)
#     good_fry_data_original = np.random.permutation(good_fry_data_original)
#     bad_fry_data_original = make_fry_data(bad_dir,label=1)
#     bad_fry_data_original = np.random.permutation(bad_fry_data_original)
#
#     good_fry_data_train,good_fry_data_test = np.split(good_fry_data_original,[int(np.floor(0.8*len(good_fry_data_original)))])
#     bad_fry_data_train,bad_fry_data_test = np.split(bad_fry_data_original,[int(np.floor(0.8*len(good_fry_data_original)))])
#
#     if Train_flag:
#         good_fry_data = good_fry_data_train
#         bad_fry_data = bad_fry_data_train
#     else:
#         good_fry_data = good_fry_data_test
#         bad_fry_data = bad_fry_data_test
#
#
#     # k = 0
#     # for i,j in zip(good_fry_data,bad_fry_data):
#     #     img_good = i[0]
#     #     img_bad = j[0]
#     #     cv2.imwrite('./nakami/%d_good.png'%(k,),img_good)
#     #     cv2.imwrite('./nakami/%d_bad.png'%(k,),img_bad)
#     #     k += 1
#
#
#     for t in range(times):
#         sys.stdout.write('.')
#         sys.stdout.flush()
#         canvas = make_canvas(csize,color)
#         H_o,W_o,_ = canvas.shape
#         #these 2 should be devided for test#
#
#
#         max = 16
#         # num = np.random.randint(0,max+1)
#
#         num = 8
#         #-*- np.random.randint(始点,終点+1,個数) -*-#
#         good_num = np.random.randint(0,len(good_fry_data),num)
#         bad_num = np.random.randint(0,len(bad_fry_data),max-num)
#
#         datas = []
#         # print 'good_num:%d'%(num,)
#         # print 'bad_num:%d'%(20-num,)
#         for i in good_num:
#             datas.append(good_fry_data[i])
#         for i in bad_num:
#             datas.append(bad_fry_data[i])
#         datas = np.random.permutation(datas)
#
#         ypoints = [40,840,1640,2440]
#         xpoints = [40,840,1640,2440]
#         bboxes = []
#         labels = []
#         mask = make_canvas(size=(1200,1200),color=0)
#
#         for i,y in enumerate(ypoints):
#             id = 0
#             for j,x in enumerate(xpoints):
#                 x_offset = np.random.randint(-30, 31) + x
#                 y_offset = np.random.randint(-30, 31) + y
#                 ind = i*len(xpoints) + j
#                 img = datas[ind][0]
#                 # xb,yb,wb,hb = datas[ind][1]
#                 label = datas[ind][2]
#                 img_rotated,_ = rotate_img(img,[0,340,20],cv2_flag=True)
#                 img_tocanvas = img_rotated.copy()
#                 img_tocanvas[np.where((img_tocanvas == [0,0,0]).all(axis=2))] = color
#                 bbox_info = make_bbox(img_rotated)
#                 mask_tmp = getMask(img_rotated)
#                 H,W,_ = img_rotated.shape
#
#                 miny = y_offset
#                 maxy = y_offset+H
#                 minx = x_offset
#                 maxx = x_offset+W
#                 # print canvas.shape
#                 # print img_rotated.shape
#                 canvas[miny:maxy,minx:maxx] = img_tocanvas
#                 mask[miny:maxy,minx:maxx] = mask_tmp
#
#                 miny = bbox_info[1] + y_offset
#                 maxy = bbox_info[1]+bbox_info[3] + y_offset
#                 minx = bbox_info[0] + x_offset
#                 maxx = bbox_info[0]+bbox_info[2] + x_offset
#
#                 bboxes.append([miny,minx,maxy,maxx])
#                 labels.append(label)
#         resized_canvas = trans_img_chainer(cv2.resize(canvas,dsize=None,fx=0.3,fy=0.3))
#         resized_mask = trans_img_chainer(cv2.resize(mask,dsize=None,fx=0.3,fy=0.3))
#
#         _,H_r,W_r = resized_canvas.shape
#         bboxes = np.array(bboxes)
#
#         resized_bboxes = chainercv.transforms.resize_bbox(bboxes, (H_o,W_o), (H_r,W_r))
#         result.append([resized_canvas,resized_bboxes,labels,resized_mask,id])
#         id += 1
#     sys.stdout.write('\n')
#     sys.stdout.flush()
#     return result

def rotate_img(img,angle_info,cv2_flag=False):
    #shape of img should be (c,h,w)#
    img_chainer = img
    if img.shape[2] == 3:
        img_chainer = trans_img_chainer(img_chainer)

    #angle_info = [start,finish,span]
    #if angle_info = [0,10,2] -> angles ganna be 0,2,4,6,8,10
    start,finish,span = angle_info# nanndokara nandomede wo spande
    finish += 1
    angles = range(start,finish,span)
    index = np.random.randint(0,len(angles))
    angle = angles[index]

    img_result = chainercv.transforms.rotate(img_chainer,angle,expand=True,fill=(0,0,0))
    _,H,W = img_result.shape
    if cv2_flag:
        img_result = trans_img_cv2(img_result)
    return img_result,(0,0,H,W),angle

def rotate_whole_image(info):
    result = []
    # angles = [-180,-150,-120,-90,-60,-30,0,30,60,90,120,150]
    angles = [0,90,180,270]
    for angle in angles:
        for image,bboxes,labels,info_with_resized in info:#どのキャンバス(iltukoiltukonogouseigazou を回転させるかやってる。
            sys.stdout.write('.')
            sys.stdout.flush()
            img_rot = chainercv.transforms.rotate(image,angle,expand=True,fill=(28,115,168))
            c,h,w = image.shape
            size=(h,w)
            bboxes_rot = []
            bboxes = np.array(bboxes)
            bbox_rot = chainercv.transforms.rotate_bbox(bboxes, angle, size)
            info_with_resized_rot = []
            for i in range(len(bboxes)):
              info_with_resized_rot.append(info_with_resized[i]+(bbox_rot[i],))

            result.append((img_rot,bbox_rot,labels,info_with_resized_rot))
    sys.stdout.write('\n')
    sys.stdout.flush()
    return result




#############ここから下が機械学習の部分

class Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, info,gpu):
        if gpu >=0:
            xp = cupy
        else:
            xp = numpy
        self.info = info
        self.img = []
        self.bboxes = []
        self.labels = []
        for entry in info:
            self.img.append(entry[0].astype(xp.float32))
            self.bboxes.append(xp.array(entry[1]).astype(xp.float32))
            self.labels.append(entry[2])

    def __len__(self):
        return len(self.info)

    def get_example(self, i):
        if i < len(self.info):
            return (self.img[i],self.bboxes[i],self.labels[i])
        i -= len(dataset)
        raise IndexError

class MaskDataset(chainer.dataset.DatasetMixin):#mask-r-cnn用にデータセットを作成するクラス(今回は使ってない)

    def __init__(self, info,gpu):
        if gpu >=0:
            xp = cupy
        else:
            xp = numpy
        self.info = info
        self.img = []
        self.bboxes = []
        self.labels = []
        self.masks = []
        self.ids = []
        self.class_ids = [0,1]
        for entry in info:
            # [resized_canvas,resized_bboxes,labels,resized_mask,id]
            self.img.append(entry[0].astype(xp.float32))
            self.bboxes.append(xp.array(entry[1]).astype(xp.float32))
            self.labels.append(entry[2])
            self.masks.append(entry[3].astype(xp.float32))
            self.ids.append(entry[4])


    def __len__(self):
        return len(self.info)

    def get_example(self, i):
        if i < len(self.info):
            return (self.img[i],self.labels[i],self.bboxes[i],self.masks[i],self.ids[i])
        i -= len(dataset)
        raise IndexError


#'a'
# a = make_composite_img()
#
# for i,(img,bboxes,labels) in enumerate(a):
#     for bbox,label in zip(bboxes,labels):
#         if label == 'good':
#             color = (0,0,255)
#         else:
#             color = (0,255,0)
#         img = cv2.rectangle(img,(bbox[1],bbox[0]),(bbox[3],bbox[2]),color,3)
#     cv2.imwrite('./gouseigazou/kukei_%d.png'%(i,),img)

def main():
    parser = argparse.ArgumentParser(
        description='ChainerCV training example: Faster R-CNN')
    parser.add_argument('--gpu', '-g', type=int, default=0) #gpuの種類
    parser.add_argument('--lr', '-l', type=float, default=1e-3) #lerning rate
    parser.add_argument('--out', '-o', default='./result_models/background_white_fry_gousei_with_rotate_flagment_images',
                        help='Output directory')
    parser.add_argument('--output_model', '-om', default='./background_white_fry_gousei_with_rotate_flagment_images/fry_gousei/model.npz',
                        help='Output model filename')
    parser.add_argument('--seed', '-s', type=int, default=0) #seed値
    parser.add_argument('--step_size', '-ss', type=int, default=50000)#学習率を調整していく幅
    parser.add_argument('--epoch', '-e', type=int, default=100) #epoch数
    parser.add_argument('--snapshot_interval', '-si', type=int, default=5) #スナップショットの間隔
    parser.add_argument('--imbalance', '-im', type=bool, default=True)#どこにも使ってない
    args = parser.parse_args()

    np.random.seed(args.seed) #コマンドライン引数で指定したシード値を設定△


    bbox_label_names = ('good','bad')#学習用△

    #train_info,test_info = make_composite_img(times=300,Train_flag=True)

    #make_composite_img2(csize=(3600,3600,3),resizef=0.7,color=(168,115,28),good_dir='./good',bad_dir='./bad',times=200,Train_flag=True,rotate_flag=True)
    train_info =make_composite_img2(good_dir='./good_train_grabcut', bad_dir='./bad_train_grabcut', times=300, Train_flag=True)
    test_info = make_composite_img2(good_dir='./good_test_grabcut', bad_dir='./bad_test_grabcut', times=20, Train_flag=True)



    #train_info,test_info = make_composite_img(times=3,Train_flag=True,color=(140,140,140))#時短用、元に戻す
    train = rotate_whole_image(train_info)#make_composite_imgで合成した画像に対して[0,90,180,270]の四回転させて、画像数を増やす
    test = rotate_whole_image(test_info)
    #train はリスト、合成画像一個につき一個情報が入っている(４つぐみで一つの情報。img_rot, bbox_rot, labels,info_with_resized_rot (rotae_whole_imageの末尾参照)train = [img_rot, bbox_rot, labels, info_resized_rot]ってイメージ
    #このinfo_with_resized_rotにさらに画像配置などの情報が入っている。(filename,resize,label,x_offset,y_offset,ind,angle,color,bbox_info,resized_bboxes,bboxes_rot)
    #filename=グラブカットによって作られたフライ単体の画像の名前
    #resize = キャンバス全体のサイズを変えた。もの#label = good or bad
    #x_offset,y_offset =元々の始点
    #ind = ある特定のキャンバスの、何個目においたかの順番△
    #angle =kakudo
    #color =背景色の色が何色かのデータ
    #bbox_info = 転したデータの中で、どこにフライが写ってるか、フライを囲む矩形情報
    #resized_bboxes =キャンバス全体をリサイズしたあとの情報
    #bboxes_rot = キャンバス全体を回転させたあとの座標

    #image_placement_info→info_with_resized(+resized_bboxes)→info_with_resized_rot(+bboxes_rot)という感じでデータの復元に必要な画像をinfo_with_resized_rotにまとめて入れた。これをjson形式で保存する。

    import json
    #trainの合成画像情報の保存
    for i in range(len(train)):#作られた合成画像ごとにtxtファイルを作成し、保存する。
      name = "train_%03d.txt"%(i,)
      total_info_for_json = []
      for region_info in train[i][3]:
         if region_info[2] == 0:
             region_info[2] == 'good'
         else:#label = 1のとき
             region_info[2] == 'bad'
         dic = {'filename': region_info[0], 'resize': region_info[1], 'label': region_info[2],
                'x_offset': region_info[3], 'y_offset': region_info[4],'ind':region_info[5],
                'angle':region_info[6], 'color':region_info[7], 'bbox_info':region_info[8],
                'resized_bboxes':region_info[9].tolist(), 'bboxes_rot':region_info[10].tolist()}#データが入った辞書型の配列を作成

         total_info_for_json.append(dic)
      file1 = open('./gousei_train_info/{}'.format(name), 'w')
      #file1 = open('./gousei_train/total_image_placement_info_train_%d'%(i,))
      json.dump(total_info_for_json, file1, ensure_ascii=False, indent=4)#json形式でtotal_info_for_jsonをfile1に保存
      file1.close()

    #testの合成画像情報の保存
    for i in range(len(test)):#作られた合成画像ごとにtxtファイルを作成し、保存する。
      name = "test_%03d.txt"%(i,)
      total_info_for_json = []
      for region_info in test[i][3]:
         if region_info[2] == 0:
             region_info[2] == 'good'
         else:#label = 1のとき
             region_info[2] == 'bad'
         dic = {'filename': region_info[0], 'resize': region_info[1], 'label': region_info[2],
                'x_offset': region_info[3], 'y_offset': region_info[4],'ind':region_info[5],
                'angle':region_info[6], 'color':region_info[7], 'bbox_info':region_info[8],
                'resized_bboxes':region_info[9].tolist(), 'bboxes_rot':region_info[10].tolist()}#データが入った辞書型の配列を作成

         total_info_for_json.append(dic)
      #file2 = open('total_image_placement_info_test', 'w')
      file2 = open('./gousei_test_info/{}'.format(name), 'w')
      json.dump(total_info_for_json, file2, ensure_ascii=False, indent=4)#名前も変える
      file1.close()



    #作成した訓練データ画像を保存するプログラム
    for i,(image,bboxes,labels, _) in enumerate(train):
        img = trans_img_cv2(image)
        cv2.imwrite('./gousei_train/%d.png'%(i,),img)#cv2.imwrite()で画像ファイルに保存
        for bbox,label in zip(bboxes,labels):
             y = bbox[0]
             x = bbox[1]
             Y = bbox[2]
             X = bbox[3]
             if label == 0: #label=0:goodのとき
                #color = (0,0,255)
                color = (0,255,0) #green
             elif label == 1: #badのとき
                color = (0,0,255)#red
                 #color = (0,255,0)
             cv2.rectangle(img,(x,y),(X,Y),color,3)
        cv2.imwrite('./gousei_train_seikai/%d.png'%(i,),img)

     #作成したテストデータ画像を保存するプログラム
    for i,(image,bboxes,labels, _) in enumerate(test):
        img = trans_img_cv2(image)
        #cv2.imwrite('./white_gousei_test/%d.png'%(i,),img)
        cv2.imwrite('./gousei_test/%d.png'%(i,),img)
        for bbox,label in zip(bboxes,labels):
            y = bbox[0]
            x = bbox[1]
            Y = bbox[2]
            X = bbox[3]
            if label == 0:#label=0:goodのとき。color=(B,G,R)
                #color = (0,0,255) #×goodのとき赤になってる
                color = (0,255,0)
            elif label == 1: #label=1:badのとき
                #color = (0,255,0) #×badのとき緑になってる
                color = (0,0,255)
            cv2.rectangle(img,(x,y),(X,Y),color,3)
        cv2.imwrite('./gousei_test_seikai/%d.png'%(i,),img)


    train_data = Dataset(train,args.gpu)#機械学習用・・・学習用にデータを整えるためにtrainをtrain_dataに変換する
    test_data = Dataset(test,args.gpu)

#ここはメモリの容量圧迫を避けるために書いたらしい△動画で言及している
    del train
    del test



    print("Generate FasterRCNNVGG16 from pretrained model 'imagenet'.")
    faster_rcnn = make_serializable_object(
        FasterRCNNVGG16,
        constructor_args = {
            'n_fg_class': len(bbox_label_names),
            'pretrained_model': 'imagenet',
        },
        template_args = {
            # Do not retrieve the pre-trained model again on generating a
            # template object for loading weights in a file.
            'n_fg_class': len(bbox_label_names),
            'pretrained_model': None,
        }
    )
    # faster_rcnn = load_npz_with_structure('./result_models/fry_gousei_with_rotate_flagment_images/model_ep30.npz')
    print("  Finished.")
    faster_rcnn.use_preset('evaluate')
    model = FasterRCNNTrainChain(faster_rcnn)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    print("Prepare optimizer.")
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    print("Prepare transformed dataset.")
    train_data = TransformDataset(train_data, Transform(faster_rcnn))
    test_data = TransformDataset(test_data, Transform(faster_rcnn))

    print("Prepare train_iter.")
    train_iter = chainer.iterators.SerialIterator(
        train_data, batch_size=1)
    print("Prepare test_iter.")
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)

    print("Prepare updater.")
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    print("Prepare trainer.")
    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(
        my_save_model, trigger=(5, 'epoch')#5エポックごとにmodelを保存
    )

    # trainer.extend(
    #     extensions.snapshot_object(
    #         model.faster_rcnn,
    #         'snapshot_model_{.updater.iteration}.npz'),
    #     trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.step_size, 'iteration'))

    log_interval = 1, 'epoch'
    plot_interval = 20, 'iteration'
    print_interval = 20, 'iteration'

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/roi_loc_loss',
         'main/roi_cls_loss',
         'main/rpn_loc_loss',#rpn=region proposal network(領域を提案するネットワーク)
         'main/rpn_cls_loss',
         'validation/main/map',
         'val/main/loss'#testの損失関数(新しく追加した)
         ]), trigger=(20,'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu), name='val')#testの損失関数の確認用

    # if extensions.PlotReport.available():
    #     trainer.extend(
    #         extensions.PlotReport(
    #             ['main/loss'],
    #             file_name='loss.png', trigger=plot_interval
    #         ),
    #         trigger=plot_interval
    #     )

    # trainer.extend(
    #     DetectionVOCEvaluator(
    #         test_iter, model.faster_rcnn, use_07_metric=True,
    #         label_names=bbox_label_names),
    #     trigger=(5000,'iteration'))

    # trainer.extend(extensions.dump_graph('main/loss'))

    print("Invoke trainer.run() .")
    import sys
    sys.stdout.flush()
    trainer.run()

    print('Save the trained model as "%s".' % (args.output_model,))
    sys.stdout.flush()
    save_npz_with_structure(args.output_model, faster_rcnn)

    #chainer.serializers.save_npz(args.out + '/trainer.npz', trainer)
    #chainer.serializers.save_npz(args.out + '/optimizer.npz', optimizer)
    #chainer.serializers.save_npz(args.out + '/updater.npz', updater)


if __name__ == '__main__':
    main()
