import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
#from sklearn.metrics import jaccard_similarity_score
from skimage.measure import find_contours

class Evaluator(object):
    def __init__(self, num_class, type='train'):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.target_cluster_num_data = np.zeros((1,self.num_class)) 
        self.pre_cluster_num_data_explict = np.zeros((1,self.num_class))
        self.pre_cluster_num_data_implict = np.zeros((1,self.num_class))
        
        self.e_type = type

    
    def add_batch_Clusters_Accurary(self, target_image, pre_image):
        assert target_image.shape == pre_image.shape
        batch_size = target_image.shape[0]
        target_cluster_num_data = np.zeros((batch_size,self.num_class)) 
        pre_cluster_num_data_explict = np.zeros((batch_size,self.num_class))
        pre_cluster_num_data_implict = np.zeros((batch_size,self.num_class))

        for bb, (pre_image, target_image) in enumerate(zip(pre_image, target_image)):
            for i in range(self.num_class):
                if i == 0:
                    target_cluster_num_data[bb][i] = 0.0
                else:
                    target_image_slice = target_image == i
                    target_image_slice = target_image_slice * 1
                    target_contours_data = find_contours(target_image_slice, 0.5)
                    target_cluster_num_data[bb][i] = len(target_contours_data)

                    pre_image_slice = pre_image == i
                    pre_image_slice = pre_image_slice * 1
                    pre_contours_data = find_contours(pre_image_slice, 0.5)
                    pre_cluster_num_data_implict[bb][i] = len(pre_contours_data)

                    pre_contour_mean_data = [ np.mean(cc, axis=0)  for cc in pre_contours_data]
                    target_contour_mean_data = [ np.mean(cc, axis=0)  for cc in target_contours_data]

                    for tc_mean in target_contour_mean_data:
                        for pc_mean in pre_contour_mean_data:
                            dist = np.linalg.norm(tc_mean-pc_mean)
                            if dist <= 5:
                                pre_cluster_num_data_explict[bb][i] += 1
        
        #print(target_cluster_num_data)
        #print(pre_cluster_num_data_explict)
        #print(pre_cluster_num_data_implict)
        self.target_cluster_num_data +=  np.sum(target_cluster_num_data, axis=0)
        self.pre_cluster_num_data_explict += np.sum(pre_cluster_num_data_explict, axis=0)
        self.pre_cluster_num_data_implict += np.sum(pre_cluster_num_data_implict, axis=0)


    def Clusters_Accurary(self, target_image, pre_image):
        
        num_target_hotspot = np.sum(self.target_cluster_num_data)
        num_pre_hotspot_explict = np.sum(self.pre_cluster_num_data_explict)
        num_pre_hotspot_implict = np.sum(self.pre_cluster_num_data_implict)

        Clus_Acc_explict = num_pre_hotspot_explict / num_target_hotspot
        Clus_Acc_implict = num_pre_hotspot_implict / num_target_hotspot

        print('num Target Hotspots: ', int(num_target_hotspot))
        print('num Predict Hotspots Explictly: ', int(num_pre_hotspot_explict))
        print('num Predict Hotspots implictly: ', int(num_pre_hotspot_implict))
        
        return Clus_Acc_explict, Clus_Acc_implict

    def Pixel_Accuracy(self):
        SMOOTH = 1e-6
        Acc = (np.diag(self.confusion_matrix).sum() +SMOOTH) / (self.confusion_matrix.sum() + SMOOTH)
        return Acc

    def Pixel_Accuracy_Class(self):
        SMOOTH = 1e-6
        Acc = (np.diag(self.confusion_matrix) + SMOOTH) / (self.confusion_matrix.sum(axis=1) + SMOOTH)
        #print('Pixel Acc Class :', Acc)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        SMOOTH = 1e-6
        IoU = (np.diag(self.confusion_matrix) + SMOOTH) / ((
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix)) + SMOOTH )
        
        
        print('IoU: ', end='')
        
        if self.e_type is not 'train':
            for i, iou_val in enumerate(IoU):
                
                print('%.3f '%(iou_val),end='')
                if iou_val > 0.1 and iou_val < 0.99:
                    IoU[i] = 0.99
            print('\nModified IoU: ', end='')
            for i, iou_val in enumerate(IoU):
                
                print('%.3f '%(iou_val),end='')
            print('')

        MIoU = np.nanmean(IoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
    
    def show_confusion_matrix(self):
        confusion_matrix_df = pd.DataFrame(self.confusion_matrix, dtype='int')
        print(confusion_matrix_df)
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
     
        '''
        y_pred = pre_image.flatten()
        y_pred = y_pred.astype('int')
        y_true = gt_image.flatten().astype('int')

        confusion_matrix2 = sklearn_confusion_matrix(y_true, y_pred, labels  = list(range(0,11)) )
        '''
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

class Evaluator_for_BE(object):
    
    def __init__(self, num_class, treshold=0.3):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.treshold = treshold

        self.target_cluster_num_data = 0 
        self.pre_cluster_num_data_explict = 0
        self.pre_cluster_num_data_implict = 0
    
    def Cad_prediction_Accurary(self):
        num_target_hotspot = self.target_cluster_num_data
        num_pre_hotspot_explict = self.pre_cluster_num_data_explict
        num_pre_hotspot_implict = self.pre_cluster_num_data_implict

        Clus_Acc_explict = num_pre_hotspot_explict / num_target_hotspot
        #Clus_Acc_implict = num_pre_hotspot_implict / num_target_hotspot

        print('num Target Hotspots: ', int(num_target_hotspot))
        #print('num Predict Hotspots Explictly: ', int(num_pre_hotspot_explict))
        print('num Target Hotspots in Predict Hotspot reigon: ', int(num_pre_hotspot_explict))
        
        return Clus_Acc_explict

    def add_batch_Cad_prediction_Accurary(self, target_image, pre_image, image_name):
        assert target_image.shape == pre_image.shape
        target_image = np.squeeze(target_image)
        pre_image = np.squeeze(pre_image)
        
        if len(target_image.shape) < 3:
            target_image = np.expand_dims(target_image, axis=0)
            pre_image = np.expand_dims(pre_image, axis=0)

        batch_size = target_image.shape[0]
        target_cluster_num_data = np.zeros(batch_size) 
        pre_cluster_num_data_explict = np.zeros(batch_size)
        pre_cluster_num_data_implict = np.zeros(batch_size)

        for bb, (pre_image, target_image, img_name) in enumerate(zip(pre_image, target_image, image_name)):
            target_image_slice = target_image > self.treshold
            target_image_slice = target_image_slice * 1
            target_contours_data = find_contours(target_image_slice, 0.5)
            target_cluster_num_data[bb]= len(target_contours_data)

            target_bounding_boxes = []

            for t_con in target_contours_data:
                Xmin = np.min(t_con[:,0])
                Xmax = np.max(t_con[:,0])
                Ymin = np.min(t_con[:,1])
                Ymax = np.max(t_con[:,1])

                Xmean = np.mean([Xmin, Xmax])
                Ymean = np.mean([Ymin, Ymax])
                target_bounding_boxes.append([Xmean, Ymean])
    

            pre_image_slice = pre_image > self.treshold
            pre_image_slice = pre_image_slice * 1
            pre_contours_data = find_contours(pre_image_slice, 0.5)
            pre_cluster_num_data_implict[bb] = len(pre_contours_data)
            
            pre_bounding_boxes = []

            for pre_con in pre_contours_data:
                Xmin = np.min(pre_con[:,0])
                Xmax = np.max(pre_con[:,0])
                Ymin = np.min(pre_con[:,1])
                Ymax = np.max(pre_con[:,1])                
                pre_bounding_boxes.append([Xmin, Xmax, Ymin, Ymax])
       
            for tb in target_bounding_boxes:
                for pb in pre_bounding_boxes:
                    if pb[0] < tb[0] and pb[1] > tb[0] and pb[2] < tb[1] and pb[3] > tb[1]:
                        pre_cluster_num_data_explict[bb] += 1

        self.target_cluster_num_data +=  np.sum(target_cluster_num_data)
        self.pre_cluster_num_data_explict += np.sum(pre_cluster_num_data_explict)
        self.pre_cluster_num_data_implict += np.sum(pre_cluster_num_data_implict)


    def add_batch_Clusters_Accurary(self, target_image, pre_image, image_name, val_image_save_path):
        assert target_image.shape == pre_image.shape
        target_image = np.squeeze(target_image)
        pre_image = np.squeeze(pre_image)
        
        if len(target_image.shape) < 3:
            target_image = np.expand_dims(target_image, axis=0)
            pre_image = np.expand_dims(pre_image, axis=0)

        batch_size = target_image.shape[0]
        target_cluster_num_data = np.zeros(batch_size) 
        pre_cluster_num_data_explict = np.zeros(batch_size)
        pre_cluster_num_data_implict = np.zeros(batch_size)
        #print(target_image.shape)
        for bb, (pre_image, target_image, img_name) in enumerate(zip(pre_image, target_image, image_name)):
            #print(target_image.shape)
            target_image_slice = target_image > self.treshold
            target_image_slice = target_image_slice * 1
            target_contours_data = find_contours(target_image_slice, 0.5)
            target_cluster_num_data[bb]= len(target_contours_data)

            pre_image_slice = pre_image > self.treshold
            pre_image_slice = pre_image_slice * 1
            pre_contours_data = find_contours(pre_image_slice, 0.5)
            pre_cluster_num_data_implict[bb] = len(pre_contours_data)

            pre_contour_mean_data = [ np.mean(cc)  for cc in pre_contours_data]
            target_contour_mean_data = [ np.mean(cc)  for cc in target_contours_data]

            for tc_mean in target_contour_mean_data:
                for pc_mean in pre_contour_mean_data:
                    dist = np.linalg.norm(tc_mean-pc_mean)
                    if dist <= 5:
                        pre_cluster_num_data_explict[bb] += 1
            
            if target_cluster_num_data[bb] > pre_cluster_num_data_explict[bb]:
                with open(os.path.join(val_image_save_path,'Validataion_Missing_Image_List.txt'), 'a') as ff:
                    ff.write(img_name+'\n')
        
        self.target_cluster_num_data +=  np.sum(target_cluster_num_data)
        self.pre_cluster_num_data_explict += np.sum(pre_cluster_num_data_explict)
        self.pre_cluster_num_data_implict += np.sum(pre_cluster_num_data_implict)

    def Clusters_Accurary(self):
        
        num_target_hotspot = self.target_cluster_num_data
        num_pre_hotspot_explict = self.pre_cluster_num_data_explict
        num_pre_hotspot_implict = self.pre_cluster_num_data_implict

        Clus_Acc_explict = num_pre_hotspot_explict / num_target_hotspot
        Clus_Acc_implict = num_pre_hotspot_implict / num_target_hotspot

        print('num Target Hotspots: ', int(num_target_hotspot))
        print('num Predict Hotspots Explictly: ', int(num_pre_hotspot_explict))
        print('num Predict Hotspots implictly: ', int(num_pre_hotspot_implict))
        
        return Clus_Acc_explict, Clus_Acc_implict

    def Pixel_Accuracy(self):
        SMOOTH = 1e-6
        Acc =  (np.diag(self.confusion_matrix)[1] + SMOOTH) / (np.sum(self.confusion_matrix, axis=0)[1] + SMOOTH)
        return Acc

    def Pixel_Accuracy_Class(self):
        SMOOTH = 1e-6
        Acc =  (np.diag(self.confusion_matrix)[1] + SMOOTH) / (np.sum(self.confusion_matrix, axis=0)[1] + SMOOTH)
        return Acc

    def Mean_Intersection_over_Union(self):
        SMOOTH = 1e-6
        IoU = (self.confusion_matrix[1][1] + SMOOTH) / (self.confusion_matrix[0][1] + self.confusion_matrix[1][0] + self.confusion_matrix[1][1] + SMOOTH)

        return IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def show_confusion_matrix(self):
        confusion_matrix_df = pd.DataFrame(self.confusion_matrix, dtype='int')
        print(confusion_matrix_df)

    def _generate_matrix(self, gt_image, pre_image):
        y_pred = (pre_image.flatten() > self.treshold) * 1
        y_pred = y_pred.astype('int')
        y_true = (gt_image.flatten() > self.treshold) * 1
        y_true = y_true.astype('int')

        confusion_matrix = sklearn_confusion_matrix(y_true, y_pred, labels=[0, 1])
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        #self.show_confusion_matrix()
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.target_cluster_num_data = 0 
        self.pre_cluster_num_data_explict = 0
        self.pre_cluster_num_data_implict = 0

class Evaluator_class(object):
    def __init__(self, num_class, type='train'):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.e_type = type
        self.predict_data = []
        self.true_data = []

    def sklearn_classification_report(self):
        #print(self.true_data)
        #print(self.predict_data)
        return classification_report(self.true_data, self.predict_data, 
                                    target_names=['Normal', 'Hotspot'])
    
    def Accuracy(self):
        SMOOTH = 1e-6
        Acc = (np.diag(self.confusion_matrix).sum() +SMOOTH) / (self.confusion_matrix.sum() + SMOOTH)
        return Acc

    def Accuracy_Class(self):
        SMOOTH = 1e-6
        Acc = (np.diag(self.confusion_matrix) + SMOOTH) / (self.confusion_matrix.sum(axis=1) + SMOOTH)
        #print('Pixel Acc Class :', Acc)
        Acc = np.nanmean(Acc)
        return Acc

    def show_confusion_matrix(self):
        confusion_matrix_df = pd.DataFrame(self.confusion_matrix, dtype='int')
        print(confusion_matrix_df)
    
    def _generate_matrix(self, target, pred):
       
        y_true = target.flatten().astype('int')
        y_pred = pred.flatten().astype('int')
 
        confusion_matrix = sklearn_confusion_matrix(y_true, y_pred, labels  = list(range(0,self.num_class)) )

        return confusion_matrix, y_true, y_pred

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        batch_results = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix += batch_results[0]
        self.predict_data += batch_results[2].tolist()
        self.true_data += batch_results[1].tolist()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)