#!/usr/bin/env python
# -*- coding: utf-8 -*- #←日本語を使うために記述。２系はデフォルトでASCIIのため日本語非対応。よってutf-8に文字コードを変更。
import cv2

import argparse
import chainer
import chainercv
import numpy as np
import sys
from timeit import default_timer as timer
from chainer_npz_with_structure import load_npz_with_structure
import os
import re




def detect_object(model, cv_image):
    raw_bboxes, raw_labels, raw_scores = model.predict([cv_image])
    #
    result = []
    result_draw = []
    for bbox, label, score in zip(raw_bboxes[0], raw_labels[0], raw_scores[0]):
        # Ensure that x_offset and y_offset are not negative.
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        region_info = {
            'y_offset': bbox[0], 'x_offset': bbox[1],
            'height': bbox[2] - bbox[0] + 1,
            'width': bbox[3] - bbox[1] + 1,
            'score': score,
            'label': label,
            'name': label_names[label],
        }
        region_draw_info = {
            'label': label,
            'y_offset': bbox[0], 'x_offset': bbox[1],
            'height': bbox[2] - bbox[0] + 1,
            'width': bbox[3] - bbox[1] + 1,
            }

        result.append(region_info)
        result_draw.append(region_draw_info)
    return result,result_draw

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detect objects in given images.',
    )
    parser.add_argument('--display', action='store_true',
                        help='display images')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--label_file', default='',
                        help = 'JSON file of label names')
    parser.add_argument('--model', default='',
                        help = 'NPZ file of a trained Faster R-CNN model')
    parser.add_argument('--output_dir', '-od', default='./result_images',
                        help='Output dir')
    parser.add_argument('filenames', metavar='FILE', nargs='+',
                        help='a filename of a image')
    #args = parser.parse_args(sys.argv)
    args = parser.parse_args()

    print('Load %s...' % (args.model,))
    sys.stdout.flush()
    from timeit import default_timer as timer
    start = timer()
    path = args.model
    model = load_npz_with_structure(path)
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    end = timer()
    print('Finished. (%f [sec])' % (end - start, ))
    sys.stdout.flush()

    if 'n_fg_class' in dir(model):
        label_names = ['label%d' % n for n in range(model.n_fg_class)]
    else:
        label_names = ['label%d' % n for n in range(model.n_class - 1)]
    if args.label_file != '':
        import json
        with open(args.label_file, 'r') as fp:
            json_data = json.load(fp)
        label_names = dict()
        for k, v in json_data.items():
            label_names[int(k)] = v

    if args.gpu >= 0:
        print('Invoke model.to_gpu().')
        sys.stdout.flush()
        gpu_device_id = args.gpu
        import cupy
        with cupy.cuda.Device(gpu_device_id):
            model.to_gpu(gpu_device_id)
        print('Finished.')
        sys.stdout.flush()

    #
    #
    ind = 0
    for filename in args.filenames:
        print('Requesting %s' % (filename, ))
        sys.stdout.flush()
        #
        img = chainercv.utils.read_image(filename)
        imgd = cv2.imread(filename)
        #
        from timeit import default_timer as timer
        start = timer()
        if args.gpu >= 0:
            with cupy.cuda.Device(gpu_device_id):
                result,result_draw = detect_object(model, img)
        else:
            result,result_draw = detect_object(model, img)
        end = timer()
        print('prediction finished. (%f [sec])' % (end - start, ))
        sys.stdout.flush()
        #

        print('regions:')
        for region in result:
            print(
                '{'+
                ', '.join(['%s: %s' % (k, v) for k, v in region.items()])
                +'}'
            )

        for region in result:
            color = (0,0,255)#左からBGRの順
            name = region['name']
            x_s = int(region['x_offset'])
            y_s = int(region['y_offset'])
            w_s = int(region['width'])
            h_s = int(region['height'])
            alpha=0.4#touka

            #cv2.rectangle(imgd, (x_s,y_s), (x_s+w_s,y_s+h_s),color, 3)#(画像?,左上の始点、右下の終点、色、線の太さ)、範囲を指定する関数
            if label_names[region['label']] == 'good':#ラベルが0:slotだったら青で囲む
            	cv2.rectangle(imgd, (x_s,y_s), (x_s+w_s,y_s+h_s),(255,0,0), 5)
            if label_names[region['label']] == 'bad':#ラベルが1:on_wallだったら赤で囲む
            	cv2.rectangle(imgd, (x_s,y_s), (x_s+w_s,y_s+h_s),(0,0,255), 5)
            if label_names[region['label']] == 'multiple_fry':#ラベルが2:multiple_fryだったらピンクで囲む
            	cv2.rectangle(imgd, (x_s,y_s), (x_s+w_s,y_s+h_s),(255,0,255), 5)
            if label_names[region['label']] == 'push':#ラベルが3:pushだったら黃で囲む
            	cv2.rectangle(imgd, (x_s,y_s), (x_s+w_s,y_s+h_s),(0,255,255), 5)
            if label_names[region['label']] == 'good':#ラベルが4:goodだったら緑で囲む
            	cv2.rectangle(imgd, (x_s,y_s), (x_s+w_s,y_s+h_s),(0,255,0), 3)
            if label_names[region['label']] == 'drop_out':#ラベルが5:drop_outだったら黒で囲む
            	cv2.rectangle(imgd, (x_s,y_s), (x_s+w_s,y_s+h_s),(0,0,0), 5)



	    #putTextはラベルを表示する関数
            cv2.putText(imgd,label_names[region['label']], (x_s,y_s+h_s), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=5)
            #(画像?、表示する文字：good:bad:on wall、文字の始点、フォント、文字の大きさ、文字の色BGR、線の太さ

            region['score'] = round(region['score'], 3)#見やすいように、少数第四位で四捨五入する
            cv2.putText(imgd, str(region['score']), (x_s+160, y_s+h_s), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255),thickness=4)
            #上のやつは判別したクラスがどの程度の確率で正しいと判断したかを表す確率scoreを画像上に表示するもの

	    #cv2.rectangle(imgd,(x_s,y_s+h_s),(x_s+w_s/2,y_s+h_s+h_s/3), (0,0,50), thickness=-1)#文字に枠をつけようとしてる途中
            #cv2.addWeighted(imgd, alpha, img, 1 - alpha, 0)

        ind = ind + 1
        filename2 = filename.replace("./", "")
        filename2 = filename2.replace(".JPG", "")
        filename2 = filename2.replace(".jpg", "")
        filename2 = filename2.replace(".png", "")
        #result_filename = ('%s/test_result_%d.jpg'% (output_dir,ind) )
        result_filename = ('%s/result_%s.jpg'%(output_dir, filename2))
        print ("this is the result:%s" % (result_filename,))
        print('filename is %s'%(filename2))
        cv2.imwrite(result_filename,imgd)
    #
    if args.display:
        print('Press any key on an image window to finish the program.')
        while True:
            k = cv2.waitKey(1000)
            if k != -1:
                cv2.destroyAllWindows()
                break
