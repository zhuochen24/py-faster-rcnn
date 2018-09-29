
#file_names = [
#		'faster_rcnn_end2end_TEST_vgg16_lininterp_finetune1_iter_70000.caffemodel.txt.2018-09-16_04-11-49',
#		'faster_rcnn_end2end_TEST_vgg16_lininterp_finetune2_iter_70000.caffemodel.txt.2018-09-16_04-20-02',
#		'faster_rcnn_end2end_TEST_vgg16_lininterp_finetune3_iter_70000.caffemodel.txt.2018-09-16_04-27-17',
#		'faster_rcnn_end2end_TEST_vgg16_lininterp_finetune4_iter_70000.caffemodel.txt.2018-09-16_04-33-52']


file_names = [
'faster_rcnn_end2end_ZF_TEST_zf_lininterp_origin_iter_70000.caffemodel.txt.2018-09-28_17-00-43',
'faster_rcnn_end2end_ZF_TEST_zf_lininterp_finetune1_iter_70000.caffemodel.txt.2018-09-28_14-03-36', 
'faster_rcnn_end2end_ZF_TEST_zf_lininterp_finetune2_iter_70000.caffemodel.txt.2018-09-28_14-07-13',
'faster_rcnn_end2end_ZF_TEST_zf_lininterp_finetune3_iter_70000.caffemodel.txt.2018-09-28_14-10-51',
'faster_rcnn_end2end_ZF_TEST_zf_lininterp_finetune4_iter_70000.caffemodel.txt.2018-09-28_14-14-29',
'faster_rcnn_end2end_ZF_TEST_zf_lininterp_finetune5_iter_70000.caffemodel.txt.2018-09-28_14-18-09',
'faster_rcnn_end2end_ZF_TEST_zf_lininterp_finetune6_iter_70000.caffemodel.txt.2018-09-28_14-21-41']

for file_name in file_names:
	with open(file_name,'r') as f:
		time = []
		for line in f.readlines():
			if 'caffe_forward' in line:
				words = line.strip('\n').split(' ')
				time.append(float(words[-1][:-1]))
		print file_name, " ", sum(time)/len(time)
