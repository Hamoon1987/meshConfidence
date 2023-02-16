import numpy as np

path = "sp_op/" + "h36m_p1" + "/" + "h36m_p1" + "_"
sp_op_all = np.load(path + 'sp_op.npy')
sp_gt_all = np.load(path + 'sp_gt.npy')
conf_all = np.load(path + 'conf.npy')
sp_op_all_occ = np.load(path + 'occ_sp_op.npy')
sp_gt_all_occ = np.load(path + 'occ_sp_gt.npy')
conf_all_occ = np.load(path + 'occ_conf.npy')

choice = np.random.choice(sp_op_all.shape[0], sp_op_all.shape[0]//2, replace=False)

sp_op_train = sp_op_all[choice]
sp_op_test = np.delete(sp_op_all, [choice], axis=0)
sp_gt_train = sp_gt_all[choice]
sp_gt_test = np.delete(sp_gt_all, [choice], axis=0)
conf_train = conf_all[choice]
conf_test = np.delete(conf_all, [choice], axis=0)

sp_op_train_occ = sp_op_all_occ[choice]
sp_op_test_occ = np.delete(sp_op_all_occ, [choice], axis=0)
sp_gt_train_occ = sp_gt_all_occ[choice]
sp_gt_test_occ = np.delete(sp_gt_all_occ, [choice], axis=0)
conf_train_occ = conf_all_occ[choice]
conf_test_occ = np.delete(conf_all_occ, [choice], axis=0)

np.save(path +'train_sp_op.npy', sp_op_train) # save
np.save(path +'train_sp_gt.npy', sp_gt_train) # save
np.save(path +'train_conf.npy', conf_train) # save
np.save(path +'test_sp_op.npy', sp_op_test) # save
np.save(path +'test_sp_gt.npy', sp_gt_test) # save
np.save(path +'test_conf.npy', conf_test) # save

np.save(path +'occ_train_sp_op.npy', sp_op_train_occ) # save
np.save(path +'occ_train_sp_gt.npy', sp_gt_train_occ) # save
np.save(path +'occ_train_conf.npy', conf_train_occ) # save
np.save(path +'occ_test_sp_op.npy', sp_op_test_occ) # save
np.save(path +'occ_test_sp_gt.npy', sp_gt_test_occ) # save
np.save(path +'occ_test_conf.npy', conf_test_occ) # save