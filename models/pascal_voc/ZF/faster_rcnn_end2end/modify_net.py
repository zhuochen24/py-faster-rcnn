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
			interpLayerName = 'relu'+convLayer.name[-1]
		if interpLayerName not in layerNames:
                    print 'layer {} to interpolate not found'.format(interpLayerName)
		interp_layer_names.append(interpLayerName)
		count+=1
                # the conv layer that its computation can be reduced
		interp_bottom_name = convLayer.name
		print '"{}",'.format(convLayer.name)
		interp_bottom_names.append(interp_bottom_name)
	print "Layers to be interpolated: ", interp_layer_names

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

				print "./experiments/scripts/interp_faster_rcnn_end2end_perlayer.sh 1 ZF pascal_voc {0}_{1}.prototxt".format(file_prefix, interpLayerName)
				# rpn outputs share the output, so no break
				#break  # should I break here? what if multiple ones using the output of this interpolation?


def gen_VIP_round_new(roundInd,layers_to_interp,file_prefix, gen_solver):
	net = caffe_pb2.NetParameter()

	fn = '{}.prototxt'.format(file_prefix)
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
			if convLayer.name == 'rpn/output':
				interpLayerName = 'rpn_relu_3x3'
			else:
				interpLayerName = 'relu'+convLayer.name[4:]
			if interpLayerName not in layerNames:
                            print 'Interpolation layer not found'
			print '"{}",'.format(interpLayerName)
			interp_layer_names.append(interpLayerName)
			count+=1
			interp_bottom_name = convLayer.name
			interp_bottom_names.append(interp_bottom_name)
	print "layers to interpolate: ", interp_layer_names

	for interp_bottom_name, interpLayerName in zip(interp_bottom_names, interp_layer_names):
		for layeridx, layer in enumerate(layers):
			if interp_bottom_name in layer.bottom and layeridx > layerNames.index(interpLayerName):
				for ind, bottomLayer in enumerate(layer.bottom):
					if bottomLayer == interp_bottom_name:
						interp_top_name = 'lininterp/'+interp_bottom_name
						interp_top_names.append(interp_top_name)
						layers[layeridx].bottom[ind] = interp_top_name
				#break   ## no break here
	outFn = './tmp_train_val/tmp_{0}_round{1}.prototxt'.format(file_prefix,roundInd)
	#print 'writing', outFn
	with open(outFn, 'w') as f:
		f.write(str(net))

	newFn = './tmp_train_val/{0}_round{1}.prototxt'.format(file_prefix,roundInd)
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

	if gen_solver:
		solverTemplateFn = './tmp_solver/solver_template.prototxt'
		solverFn = './tmp_solver/solver_round{}.prototxt'.format(roundInd)
		with open(solverTemplateFn) as tempf:
			with open(solverFn, 'w') as f:
				f.write('train_net: "models/pascal_voc/ZF/faster_rcnn_end2end/tmp_train_val/train_all_round{}.prototxt"\n'.format(roundInd))
				for line in tempf.readlines():
					f.write(line)
				f.write('snapshot_prefix: "zf_lininterp_finetune{}"'.format(roundInd))

		print "./experiments/scripts/interp_faster_rcnn_end2end_finetune.sh 0 ZF pascal_voc train_all_round{0}.prototxt test_all_round{0}.prototxt".format(roundInd)

########################################========= main ===========###############################
all_layers_to_interp=[
	"origin",
	"conv1",
	"conv2",
	"conv3",
	"conv4",
	"conv5",
	"rpn/output"
]
for roundInd in range(1,7):
	if roundInd == 1:
		###### for round 1
		layers_to_interp = [all_layers_to_interp[i] for i in [1]]
	elif roundInd == 2:
		###### for round 2
		layers_to_interp = [all_layers_to_interp[i] for i in [1,6]]
	elif roundInd == 3:
		###### for round 3
		layers_to_interp = [all_layers_to_interp[i] for i in [1,5,6]]
	elif roundInd == 4:
		###### for round 4
		layers_to_interp = [all_layers_to_interp[i] for i in [1,3,5,6]]
	elif roundInd == 5:
		###### for round 4
		layers_to_interp = [all_layers_to_interp[i] for i in [1,2,3,5,6]]
	elif roundInd == 6:
		###### for round 4
		layers_to_interp = [all_layers_to_interp[i] for i in [1,2,3,4,5,6]]
	#gen_VIP_round_new(roundInd,layers_to_interp, 'train_all', gen_solver=True)
	gen_VIP_round_new(roundInd,layers_to_interp, 'test_all', gen_solver=False)

#gen_VIP_perlayer_new('train_all')
#gen_VIP_perlayer_new('test_all')

