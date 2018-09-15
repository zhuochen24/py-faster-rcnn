from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf


# generate the train_val_prototxt for each conv layer being interpolated individually.
# for testing how robust each layer is to interpolation
def gen_VIP_perlayer_new(file_prefix):
	net = caffe_pb2.NetParameter()

        fn = '{}.prototxt'.format(file_prefix)
	with open(fn) as f:
		s = f.read()
		txtf.Merge(s, net)

	layers = [l for l in net.layer]
	layerNames = [l.name for l in net.layer]
	reluLayers = [l for l in net.layer if l.type == 'ReLU']
	convLayers = [l for l in net.layer if l.type == 'Convolution']
	count=0
	interp_bottom_names = []
	interp_layer_names = []
	for convLayer in convLayers:
                #find after which layer to insert interpolate
		if convLayer.name == "rpn_cls_score" or convLayer.name == "rpn_bbox_pred":
			continue
		elif convLayer.name == 'rpn/output':
			interpLayerName = 'rpn_relu_3x3'
		else:
			interpLayerName = 'relu'+convLayer.name[4:]
		if interpLayerName not in layerNames:
                    print 'layer {} to interpolate not found'.format(interpLayerName)
		interp_layer_names.append(interpLayerName)
		count+=1
                # the conv layer that its computation can be reduced
		interp_bottom_name = convLayer.name
		print '"{}",'.format(convLayer.name)
		interp_bottom_names.append(interp_bottom_name)

        # find each conv layer to interpolate and store independently
	for interp_bottom_name, interpLayerName in zip(interp_bottom_names, interp_layer_names):
		net = caffe_pb2.NetParameter()
		fn = '{}.prototxt'.format(file_prefix)
		with open(fn) as f:
			s = f.read()
			txtf.Merge(s, net)

		layers = [l for l in net.layer]
		layerNames = [l.name for l in net.layer]
		tmp_ind = layerNames.index(interp_bottom_name)
		convLayer = net.layer[tmp_ind]
                # double the stride of the conv layer due to interpolation
		convLayer.convolution_param.stride[0] = 2*convLayer.convolution_param.stride[0]
		for layeridx, layer in enumerate(layers):
			if interp_bottom_name in layer.bottom and layeridx > layerNames.index(interpLayerName):
                                # change the bottom to lininterp+interp_bottom_name of the layer that will be after the interpolation layer (its bottom was interp_bottom_name but now interpolated)
				for ind, bottomLayer in enumerate(layer.bottom):
					if bottomLayer == interp_bottom_name:
						interp_top_name = 'lininterp/'+interp_bottom_name
						layers[layeridx].bottom[ind] = interp_top_name
				outFn = './tmp_train_val/tmp_{0}_{1}.prototxt'.format(file_prefix,interpLayerName)
				#print 'writing', outFn
				with open(outFn, 'w') as f:
					f.write(str(net))

				newFn = './tmp_train_val/{0}_{1}.prototxt'.format(file_prefix, interpLayerName)
				with open(outFn) as f:
					with open(newFn,'w') as newf:
						flag = 0
						for line in f.readlines():
							if interpLayerName in line:
								flag=1
							if flag == 1 and 'layer' in line:
								flag = 0
								newf.write('layer {\n')
								newf.write('  type: "Interp"\n')
								newf.write('  name: "{}"\n'.format(interp_top_name))
								newf.write('  top: "{}"\n'.format(interp_top_name))
								newf.write('  bottom: "{}"\n'.format(interp_bottom_name))
								newf.write('}\n')
							newf.write(line)

				print "./experiments/scripts/interp_faster_rcnn_end2end_perlayer.sh 2 VGG16 pascal_voc {0}_{1}.prototxt".format(file_prefix, interpLayerName)
				# rpn outputs share the output, so no break
				#break  # should I break here? what if multiple ones using the output of this interpolation?

def gen_VIP_round(roundInd,layers_to_interp):
	net = caffe_pb2.NetParameter()

	fn = 'train_val_resnet50.prototxt'
	with open(fn) as f:
		s = f.read()
		txtf.Merge(s, net)

	layers = [l for l in net.layer]
	layerNames = [l.name for l in net.layer]
	convLayers = [l for l in net.layer if l.type == 'Convolution']
	#convLayerNames = [l.name for l in net.layer if l.type == 'Convolution']

	count=0
	interp_bottom_names = []
	interp_top_names = []
	interp_layer_names = []
	for convLayer in convLayers:
		if convLayer.name in layers_to_interp:
			convLayer.convolution_param.stride[0] = 2*convLayer.convolution_param.stride[0]
			interpLayerName = convLayer.name+'_relu'
			if interpLayerName not in layerNames:
				interpLayerName = 'scale'+convLayer.name.strip('res')
			print '"{}",'.format(interpLayerName)
			interp_layer_names.append(interpLayerName)
			count+=1
			interp_bottom_name = convLayer.name
			interp_bottom_names.append(interp_bottom_name)

	for interp_bottom_name, interpLayerName in zip(interp_bottom_names, interp_layer_names):
		for layeridx, layer in enumerate(layers):
			if interp_bottom_name in layer.bottom and layeridx > layerNames.index(interpLayerName):
				for ind, bottomLayer in enumerate(layer.bottom):
					if bottomLayer == interp_bottom_name:
						interp_top_name = 'lininterp/'+interp_bottom_name
						interp_top_names.append(interp_top_name)
						layers[layeridx].bottom[ind] = interp_top_name
				#break   ## no break here
	outFn = './tmp_train_val/tmp_train_val_resnet50_round{}.prototxt'.format(roundInd)
	#print 'writing', outFn
	with open(outFn, 'w') as f:
		f.write(str(net))

	newFn = './tmp_train_val/train_val_resnet50_round{}.prototxt'.format(roundInd)
	with open(outFn) as f:
		with open(newFn,'w') as newf:
			flag = 0
			dummyExist = 0
			for line in f.readlines():
				for interpIdx, interpLayerName in enumerate(interp_layer_names):
					if interpLayerName in line:
						if '5a' in interpLayerName or '5b' in interpLayerName or '5c' in interpLayerName:
							flag = 2
						else:
							flag=1
						interp_top_name = interp_top_names[interpIdx]
						interp_bottom_name = interp_bottom_names[interpIdx]
						break
					if flag == 2 and 'layer' in line:
						flag = 0
						if not dummyExist:
							newf.write('layer {\n')
							newf.write('  type: "DummyData"\n')
							newf.write('  name: "{}"\n'.format('onlyone_dummy'))
							newf.write('  top: "{}"\n'.format('onlyone_dummy'))
							newf.write('  dummy_data_param{\n')
							newf.write('    num: 50\n')
							newf.write('    channels: 512\n')
							newf.write('    width: 7\n')
							newf.write('    height: 7\n')
							newf.write('    }\n')
							newf.write('}\n')
							dummyExist = 1
						newf.write('layer {\n')
						newf.write('  type: "Interp"\n')
						newf.write('  name: "{}"\n'.format(interp_top_name+'_tmp'))
						newf.write('  top: "{}"\n'.format(interp_top_name+'_tmp'))
						newf.write('  bottom: "{}"\n'.format(interp_bottom_name))
						newf.write('}\n')


						newf.write('layer {\n')
						newf.write('  type: "Crop"\n')
						newf.write('  name: "{}"\n'.format(interp_top_name))
						newf.write('  bottom: "{}"\n'.format(interp_top_name+'_tmp'))
						newf.write('  bottom: "{}"\n'.format('onlyone_dummy'))
						newf.write('  top: "{}"\n'.format(interp_top_name))
						newf.write('  crop_param{\n')
						newf.write('    axis: 2\n')
						newf.write('    offset: 0\n')
						newf.write('    }\n')
						newf.write('}\n')
					if flag == 1 and 'layer' in line:
						flag = 0
						newf.write('layer {\n')
						newf.write('  type: "Interp"\n')
						newf.write('  name: "{}"\n'.format(interp_top_name))
						newf.write('  top: "{}"\n'.format(interp_top_name))
						newf.write('  bottom: "{}"\n'.format(interp_bottom_name))
						newf.write('}\n')
						break
					elif flag==1:
						break
				newf.write(line)

	print './build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round{0}.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round{0}_interpCUDAv3.out'.format(roundInd)
	print "./build/tools/caffe train --solver=models/resnet/solver_resnet50_round{0}.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 2>&1 | tee ./resnet_results/resnet50_lininterp_round{0}.out".format(roundInd)


def gen_VIP_round_new(roundInd,layers_to_interp):
	net = caffe_pb2.NetParameter()

	fn = 'ALL_CNN_C_train_val.prototxt'
	with open(fn) as f:
		s = f.read()
		txtf.Merge(s, net)

	layers = [l for l in net.layer]
	layerNames = [l.name for l in net.layer]
	convLayers = [l for l in net.layer if l.type == 'Convolution']
	#convLayerNames = [l.name for l in net.layer if l.type == 'Convolution']

	count=0
	interp_bottom_names = []
	interp_top_names = []
	interp_layer_names = []
	for convLayer in convLayers:
		if convLayer.name in layers_to_interp:
			convLayer.convolution_param.stride[0] = 2*convLayer.convolution_param.stride[0]
                        interpLayerName = 'relu'+convLayer.name[-1]
			if interpLayerName not in layerNames:
                            print 'Interpolation layer not found'
			#print '"{}",'.format(interpLayerName)
			interp_layer_names.append(interpLayerName)
			count+=1
			interp_bottom_name = convLayer.name
			interp_bottom_names.append(interp_bottom_name)

	for interp_bottom_name, interpLayerName in zip(interp_bottom_names, interp_layer_names):
		for layeridx, layer in enumerate(layers):
			if interp_bottom_name in layer.bottom and layeridx > layerNames.index(interpLayerName):
				for ind, bottomLayer in enumerate(layer.bottom):
					if bottomLayer == interp_bottom_name:
						interp_top_name = 'lininterp/'+interp_bottom_name
						interp_top_names.append(interp_top_name)
						layers[layeridx].bottom[ind] = interp_top_name
				#break   ## no break here
	outFn = './tmp_train_val/tmp_train_val_ALLCNNC_round{}.prototxt'.format(roundInd)
	#print 'writing', outFn
	with open(outFn, 'w') as f:
		f.write(str(net))

	newFn = './tmp_train_val/train_val_ALLCNNC_round{}.prototxt'.format(roundInd)
	with open(outFn) as f:
		with open(newFn,'w') as newf:
			flag = 0
			dummyExist = 0
			for line in f.readlines():
				for interpIdx, interpLayerName in enumerate(interp_layer_names):
					if interpLayerName in line:
						flag=1
						interp_top_name = interp_top_names[interpIdx]
						interp_bottom_name = interp_bottom_names[interpIdx]
						break
					if flag == 1 and 'layer' in line:
						flag = 0
						newf.write('layer {\n')
						newf.write('  type: "Interp"\n')
						newf.write('  name: "{}"\n'.format(interp_top_name))
						newf.write('  top: "{}"\n'.format(interp_top_name))
						newf.write('  bottom: "{}"\n'.format(interp_bottom_name))
						newf.write('}\n')
						break
					elif flag==1:
						break
				newf.write(line)

	solverTemplateFn = './tmp_solver/ALL_CNN_C_solver_template.prototxt'
	solverFn = './tmp_solver/solver_ALLCNNC_round{}.prototxt'.format(roundInd)
	with open(solverTemplateFn) as tempf:
		with open(solverFn, 'w') as f:
			f.write('net: "models/ALL-CNN/tmp_train_val/train_val_ALLCNNC_round{}.prototxt"\n'.format(roundInd))
			for line in tempf.readlines():
				f.write(line)
			f.write('snapshot_prefix: "models/ALL-CNN/snapshots/ALLCNNC_lininterp_finetune{}_CUDAv5"'.format(roundInd))

	print './build/tools/caffe time --model=models/ALL-CNN/tmp_train_val/train_val_ALLCNNC_round{0}.prototxt  -gpu 1  2>&1 | tee  ./allcnn_results/time_ALLCNNC_lininterp_round{0}_interpCUDAv5.out'.format(roundInd)
	print "./build/tools/caffe train --solver=models/ALL-CNN/tmp_solver/solver_ALLCNNC_round{0}.prototxt --weights=models/ALL-CNN/snapshots/ALLCNNC_lininterp_finetune{1}_CUDAv5.caffemodel -gpu 1 2>&1 | tee ./allcnn_results/ALLCNNC_lininterp_round{0}_CUDAv5.out".format(roundInd, roundInd-1)

########################################========= main ===========###############################
#`all_layers_to_interp=[
#`	"origin",
#`	"conv1",
#`	"conv2",
#`	"conv3",
#`	"conv4",
#`	"conv5",
#`	"conv6",
#`	"conv7",
#`	"conv8",
#`	"conv9"
#`]
#`roundInd = 1
#`round3Elem = [27,28,35,45,4,5,26,1,14,15,3,41,51,22]
#`if roundInd == 1:
#`	###### for round 1
#`	layers_to_interp = [all_layers_to_interp[i] for i in [10,11,17,18,19,20,23,24,29,30,31,32,33,34,36,37,38,39,40,42,43]]
#`elif roundInd == 2:
#`	###### for round 2
#`	layers_to_interp = [all_layers_to_interp[i] for i in [10,11,17,18,19,20,23,24,29,30,31,32,33,34,36,37,38,39,40,42,43,21,46,47,48,49,50,52,53]]
#`elif roundInd == 3:
#`	###### for round 3
#`	for perlayer in range(len(round3Elem)):
#`		layers_to_interp = [all_layers_to_interp[i] for i in [10,11,17,18,19,20,23,24,29,30,31,32,33,34,36,37,38,39,40,42,43,21,46,47,48,49,50,52,53]+round3Elem[:perlayer+1]]
#`		gen_VIP_round_new(roundInd*10+perlayer+1,layers_to_interp)

gen_VIP_perlayer_new('train_all')
gen_VIP_perlayer_new('test_all')

