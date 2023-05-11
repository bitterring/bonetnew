import tensorflow as tf
import os
import shutil
from helper_net import Ops as Ops

class BoNet:
	def __init__(self, configs):
		self.points_cc = configs.points_cc
		self.sem_num = configs.sem_num
		self.bb_num = configs.ins_max_num

	def creat_folders(self, name='log', re_train=False):
		self.train_mod_dir = './'+name+'/train_mod/'
		self.train_sum_dir = './'+name+'/train_sum/'
		self.test_sum_dir = './'+name+'/test_sum/'
		print ("re_train:", re_train)
		def tp(path):
			if os.path.exists(path):
				if re_train:
					print (path, ": files kept!")
				else:
					shutil.rmtree(path)
					os.makedirs(path)
					print (path, ': deleted and then created!')
			else:
				os.makedirs(path)
				print (path, ': created!')
		tp(self.test_sum_dir)
		tp(self.train_sum_dir)
		tp(self.train_mod_dir)

	######  1. backbone + sem
	def backbone_pointnet(self, X_pc, is_train):
		[_, _, points_cc] = X_pc.get_shape()
		points_num = tf.shape(input=X_pc)[1]
		X_pc = tf.reshape(X_pc, [-1, points_num, int(points_cc), 1])
		with tf.device('/gpu:0'):
			l1 = Ops.xxlu(Ops.conv2d(X_pc, k=(1, points_cc), out_c=64, str=1, pad='VALID', name='l1'), label='lrelu')
			l2 = Ops.xxlu(Ops.conv2d(l1, k=(1, 1), out_c=64, str=1, pad='VALID', name='l2'), label='lrelu')
			l3 = Ops.xxlu(Ops.conv2d(l2, k=(1, 1), out_c=64, str=1, pad='VALID', name='l3'), label='lrelu')
			l4 = Ops.xxlu(Ops.conv2d(l3, k=(1, 1), out_c=128, str=1, pad='VALID', name='l4'), label='lrelu')
			l5 = Ops.xxlu(Ops.conv2d(l4, k=(1, 1), out_c=1024, str=1, pad='VALID', name='l5'), label='lrelu')
			global_features = tf.reduce_max(input_tensor=l5, axis=1, name='maxpool')
			global_features = tf.reshape(global_features, [-1, int(l5.shape[-1])])
			point_features = tf.reshape(l5, [-1, points_num, int(l5.shape[-1])])

		####  sem
			g1 = Ops.xxlu(Ops.fc(global_features, out_d=256, name='semg1'), label='lrelu')
			g2 = Ops.xxlu(Ops.fc(g1, out_d=128, name='semg2'), label='lrelu')
			sem1 = tf.tile(g2[:,None,None,:], [1, points_num, 1, 1])
			sem1 = tf.concat([l5, sem1], axis=-1)
			sem1 = Ops.xxlu(Ops.conv2d(sem1, k=(1,1), out_c=512, str=1, pad='VALID', name='sem1'), label='lrelu')
			sem2 = Ops.xxlu(Ops.conv2d(sem1, k=(1, 1), out_c=256, str=1, pad='VALID', name='sem2'), label='lrelu')
			sem3 = Ops.xxlu(Ops.conv2d(sem2, k=(1, 1), out_c=128, str=1, pad='VALID', name='sem3'), label='lrelu')
			sem3 = Ops.dropout(sem3, keep_prob=0.5, is_train=is_train, name='sem3_dropout')
			sem4 = Ops.conv2d(sem3, k=(1, 1), out_c=self.sem_num, str=1, pad='VALID', name='sem4')
			sem4 = tf.reshape(sem4, [-1, points_num, self.sem_num])
			self.y_psem_logits = sem4
			y_sem_pred = tf.nn.softmax(self.y_psem_logits, name='y_sem_pred')

		return point_features, global_features, y_sem_pred

	def backbone_pointnet2(self, X_pc, is_train=None):
		import helper_pointnet2 as pnet2
		points_num = tf.shape(input=X_pc)[1]
		l0_xyz = X_pc[:,:,0:3]
		l0_points = X_pc[:,:,3:9]
		l0_xyz_expand=tf.expand_dims(l0_xyz,2)
		#ndata=l0_xyz.get_shape()[1].value
		with tf.device('/gpu:0'):
			R=tf.tile(tf.expand_dims(tf.constant([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]]),0),[4096,1,1])
			R=tf.tile(tf.expand_dims(R,0),[4,1,1,1])
		#print(R)
			l0_xyz=tf.squeeze(tf.matmul(l0_xyz_expand,R))
		#print(l0_xyz)

			l1_xyz, l1_points, l1_indices = pnet2.pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32,
				mlp=[32, 32, 64], mlp2=None, group_all=False, is_training=None, bn_decay=None, scope='layer1')
			#print(l1_xyz)
			l2_xyz, l2_points, l2_indices = pnet2.pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=64,
				mlp=[64, 64, 128], mlp2=None, group_all=False, is_training=None, bn_decay=None, scope='layer2')
			l3_xyz, l3_points, l3_indices = pnet2.pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=128,
				mlp=[128, 128, 256], mlp2=None, group_all=False, is_training=None, bn_decay=None, scope='layer3')
			l4_xyz, l4_points, l4_indices = pnet2.pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None,
				mlp=[256, 256, 512], mlp2=None, group_all=True, is_training=None, bn_decay=None, scope='layer4')

		# Feature Propagation layers
			l3_points = pnet2.pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], is_training=None, bn_decay=None, scope='fa_layer1')
			l2_points = pnet2.pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], is_training=None, bn_decay=None,scope='fa_layer2')
			l1_points = pnet2.pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training=None, bn_decay=None,scope='fa_layer3')
			l0_points = pnet2.pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz, l0_points], axis=-1),
		            l1_points,[128, 128, 128, 128], is_training=None, bn_decay=None, scope='fa_layer4')
			global_features = tf.reshape(l4_points, [-1, 512])
			point_features = l0_points

		# sem
			l0_points = l0_points[:,:,None,:]
			sem1 = Ops.xxlu(Ops.conv2d(l0_points, k=(1, 1), out_c=128, str=1, pad='VALID', name='sem1'), label='lrelu')
			sem2 = Ops.xxlu(Ops.conv2d(sem1, k=(1, 1), out_c=64, str=1, pad='VALID', name='sem2'), label='lrelu')
			sem2 = Ops.dropout(sem2, keep_prob=0.5, is_train=is_train, name='sem2_dropout')
			sem3 = Ops.conv2d(sem2, k=(1, 1), out_c=self.sem_num, str=1, pad='VALID', name='sem3')
			sem3 = tf.reshape(sem3, [-1, points_num, self.sem_num])
			self.y_psem_logits = sem3
			y_sem_pred = tf.nn.softmax(self.y_psem_logits, name='y_sem_pred')

		return point_features, global_features, y_sem_pred

	######  2. bbox
	def bbox_net(self, global_features):
		with tf.device('/gpu:0'):
			b1 = Ops.xxlu(Ops.fc(global_features, out_d= 512, name='b1'), label='lrelu')
			b2 = Ops.xxlu(Ops.fc(b1, out_d= 256, name='b2'), label='lrelu')

		#### sub branch 1
			b3 = Ops.xxlu(Ops.fc(b2, out_d=256, name='b3'), label='lrelu')
			bbvert = Ops.fc(b3, out_d=self.bb_num * 2 * 3, name='bbvert')
			bbvert = tf.reshape(bbvert, [-1, self.bb_num, 2, 3])
			points_min = tf.reduce_min(input_tensor=bbvert, axis=-2)[:, :, None, :]
			points_max = tf.reduce_max(input_tensor=bbvert, axis=-2)[:, :, None, :]
			y_bbvert_pred = tf.concat([points_min, points_max], axis=-2, name='y_bbvert_pred')

		#### sub branch 2
			b4 = Ops.xxlu(Ops.fc(b2, out_d=256, name='b4'), label='lrelu')
			y_bbscore_pred = tf.sigmoid(Ops.fc(b4, out_d=self.bb_num * 1, name='y_bbscore_pred'))

		return y_bbvert_pred, y_bbscore_pred

	######  3. pmask
	def pmask_net(self, point_features, global_features, bbox, bboxscore):
		with tf.device('/gpu:0'):
			p_f_num = int(point_features.shape[-1])
			p_num = tf.shape(input=point_features)[1]
			bb_num = int(bbox.shape[1])

			global_features = tf.tile(Ops.xxlu(Ops.fc(global_features, out_d=256, name='down_g1'), label='lrelu')[:,None,None,:], [1, p_num, 1, 1])
			point_features = Ops.xxlu(Ops.conv2d(point_features[:,:,:,None],k=(1, p_f_num), out_c=256, str=1,name='down_p1',pad='VALID'), label='lrelu')
			point_features = tf.concat([point_features, global_features], axis=-1)
			point_features = Ops.xxlu(Ops.conv2d(point_features, k=(1,int(point_features.shape[-2])), out_c=128, str=1, pad='VALID', name='down_p2'), label='lrelu')
			point_features = Ops.xxlu(Ops.conv2d(point_features, k=(1, int(point_features.shape[-2])), out_c=128, str=1, pad='VALID',name='down_p3'), label='lrelu')
			point_features = tf.squeeze(point_features, axis=-2)

			bbox_info = tf.tile(tf.concat([tf.reshape(bbox, [-1, bb_num, 6]), bboxscore[:,:,None]],axis=-1)[:,:,None,:], [1,1,p_num,1])
			pmask0 = tf.tile(point_features[:,None,:,:], [1, bb_num, 1, 1])
			pmask0 = tf.concat([pmask0, bbox_info], axis=-1)
			pmask0 = tf.reshape(pmask0, [-1, p_num, int(pmask0.shape[-1]), 1])

			pmask1 = Ops.xxlu(Ops.conv2d(pmask0, k=(1,int(pmask0.shape[-2])), out_c=64, str=1, pad='VALID', name='pmask1'), label='lrelu')
			pmask2 = Ops.xxlu(Ops.conv2d(pmask1, k=(1, 1), out_c=32, str=1, pad='VALID', name='pmask2'),label='lrelu')
			pmask3 = Ops.conv2d(pmask2, k=(1,1), out_c=1, str=1, pad='VALID', name='pmask3')
			pmask3 = tf.reshape(pmask3, [-1, bb_num, p_num])

			y_pmask_logits = pmask3
			y_pmask_pred = tf.nn.sigmoid(y_pmask_logits, name='y_pmask_pred')

		return y_pmask_pred

	######
	def build_graph(self, GPU='0'):
		tf.ConfigProto(log_device_placement=True)
		#######   1. define inputs
		tf.compat.v1.disable_eager_execution()
		self.X_pc = tf.compat.v1.placeholder(shape=[4, 4096, self.points_cc], dtype=tf.float32, name='X_pc')
		self.Y_bbvert = tf.compat.v1.placeholder(shape=[None, self.bb_num, 2, 3], dtype=tf.float32, name='Y_bbvert')
		self.Y_pmask = tf.compat.v1.placeholder(shape=[None, self.bb_num, None], dtype=tf.float32, name='Y_pmask')
		self.Y_psem = tf.compat.v1.placeholder(shape=[None, None, self.sem_num], dtype=tf.float32, name='Y_psem')
		self.is_train = tf.compat.v1.placeholder(dtype=tf.bool, name='is_train')
		self.lr = tf.compat.v1.placeholder(dtype=tf.float32, name='lr')

		#######  2. define networks, losses
		with tf.compat.v1.variable_scope('backbone'):
			#self.point_features, self.global_features, self.y_psem_pred = self.backbone_pointnet(self.X_pc, self.is_train)
			self.point_features, self.global_features, self.y_psem_pred = self.backbone_pointnet2(self.X_pc, self.is_train)

			### loss
			with tf.device('/gpu:0'):
				self.psemce_loss = Ops.get_loss_psem_ce(self.y_psem_logits, self.Y_psem)
			self.sum_psemce_loss = tf.compat.v1.summary.scalar('psemce_loss', self.psemce_loss)

		with tf.compat.v1.variable_scope('bbox'):
			self.y_bbvert_pred_raw, self.y_bbscore_pred_raw = self.bbox_net(self.global_features)
			#### association, only used for training
			bbox_criteria = 'use_all_ce_l2_iou'
			with tf.device('/gpu:0'):
				self.y_bbvert_pred, self.pred_bborder = Ops.bbvert_association(self.X_pc,  self.y_bbvert_pred_raw, self.Y_bbvert, label=bbox_criteria)
				self.y_bbscore_pred = Ops.bbscore_association(self.y_bbscore_pred_raw, self.pred_bborder)

			### loss
			with tf.device('/gpu:0'):
				self.bbvert_loss, self.bbvert_loss_l2, self.bbvert_loss_ce, self.bbvert_loss_iou = \
					Ops.get_loss_bbvert(self.X_pc, self.y_bbvert_pred, self.Y_bbvert, label=bbox_criteria)
				self.bbscore_loss = Ops.get_loss_bbscore(self.y_bbscore_pred, self.Y_bbvert)
				self.sum_bbox_vert_loss = tf.compat.v1.summary.scalar('bbvert_loss', self.bbvert_loss)
				self.sum_bbox_vert_loss_l2 = tf.compat.v1.summary.scalar('bbvert_loss_l2', self.bbvert_loss_l2)
				self.sum_bbox_vert_loss_ce = tf.compat.v1.summary.scalar('bbvert_loss_ce', self.bbvert_loss_ce)
				self.sum_bbox_vert_loss_iou = tf.compat.v1.summary.scalar('bbvert_loss_iou', self.bbvert_loss_iou)
				self.sum_bbox_score_loss = tf.compat.v1.summary.scalar('bbscore_loss', self.bbscore_loss)

		with tf.compat.v1.variable_scope('pmask'):
			with tf.device('/gpu:0'):
				self.y_pmask_pred = self.pmask_net(self.point_features, self.global_features, self.y_bbvert_pred, self.y_bbscore_pred)

			### loss
				self.pmask_loss = Ops.get_loss_pmask(self.X_pc, self.y_pmask_pred, self.Y_pmask)
				self.sum_pmask_loss = tf.compat.v1.summary.scalar('pmask_loss', self.pmask_loss)

		with tf.compat.v1.variable_scope('pmask', reuse=True):
			#### during testing, no need to associate, use unordered predictions
			with tf.device('/gpu:0'):
				self.y_pmask_pred_raw = self.pmask_net(self.point_features, self.global_features, self.y_bbvert_pred_raw, self.y_bbscore_pred_raw)

		######   3. define optimizers
		with tf.device('/gpu:0'):
			var_backbone = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith('backbone') and not var.name.startswith('backbone/sem')]
			var_sem = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith('backbone/sem')]
			var_bbox = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith('bbox')]
			var_pmask = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith('pmask')]

			end_2_end_loss = self.bbvert_loss + self.bbscore_loss  + self.pmask_loss + self.psemce_loss
		with tf.device('/gpu:0'):
			self.optim = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(end_2_end_loss, var_list = var_bbox+var_pmask +var_backbone+ var_sem)

		######   4. others
		print(Ops.variable_count())
		self.saver = tf.compat.v1.train.Saver(max_to_keep=20)
		config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
		config.gpu_options.visible_device_list = GPU
		config.gpu_options.allow_growth = True
		self.sess = tf.compat.v1.Session(config=config)
		self.sum_writer_train = tf.compat.v1.summary.FileWriter(self.train_sum_dir, self.sess.graph)
		self.sum_write_test = tf.compat.v1.summary.FileWriter(self.test_sum_dir)
		self.sum_merged = tf.compat.v1.summary.merge_all()

		path = self.train_mod_dir
		if os.path.isfile(path + 'model.cptk.data-00000-of-00001'):
			print ("restoring saved model")
			self.saver.restore(self.sess, path + 'model.cptk')
		else:
			print ("model not found, all weights are initilized")
			self.sess.run(tf.compat.v1.global_variables_initializer())

		return 0
	
if __name__=='__main__':
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ## specify the GPU to use

	#from main_3D_BoNet import BoNet
	from helper_data_s3dis import Data_Configs as Data_Configs

	configs = Data_Configs()
	net = BoNet(configs = configs)
	net.creat_folders(name='log', re_train=False)
	net.build_graph()

	####
	from helper_data_s3dis import Data_S3DIS as Data
	train_areas =['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
	test_areas =['Area_5']

	dataset_path = './data_s3dis/'
	data = Data(dataset_path, train_areas, test_areas, train_batch_size=4)
	l_rate = max(0.0005/(2**(0//20)), 0.00001)

	data.shuffle_train_files(0)
	total_train_batch_num = data.total_train_batch_num
	with tf.device('/gpu:0'):
		bat_pc, _, _, bat_psem_onehot, bat_bbvert, bat_pmask = data.load_train_next_batch()
		_, ls_psemce, ls_bbvert_all, ls_bbvert_l2, ls_bbvert_ce, ls_bbvert_iou, ls_bbscore, ls_pmask = net.sess.run([
				net.optim, net.psemce_loss, net.bbvert_loss, net.bbvert_loss_l2, net.bbvert_loss_ce, net.bbvert_loss_iou,net.bbscore_loss, net.pmask_loss],
				feed_dict={net.X_pc:bat_pc[:, :, 0:9], net.Y_bbvert:bat_bbvert, net.Y_pmask:bat_pmask, net.Y_psem:bat_psem_onehot, net.lr:l_rate, net.is_train:True})

	
	print ('ep', 0, 'i', 0, 'psemce', ls_psemce, 'bbvert', ls_bbvert_all, 'l2', ls_bbvert_l2, 'ce', ls_bbvert_ce, 'siou', ls_bbvert_iou, 'bbscore', ls_bbscore, 'pmask', ls_pmask)
