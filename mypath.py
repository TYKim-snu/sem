class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset =='samsung_CAD_BE_crop_256_NormalRatio_1':
            return '/data2/0_Samsung_Data/3_Train_Data/Hotspot_prediction_using_CAD/DataSet_1_Size_256x256_NormalRatio_1'
        elif dataset =='samsung_CAD_BE_crop_512_NormalRatio_1':
            return '/data2/0_Samsung_Data/3_Train_Data/Hotspot_prediction_using_CAD/DataSet_1_Size_512x512_NormalRatio_1'
        elif dataset =='samsung_CAD_BE_1024':
            return '/data2/0_Samsung_Data/3_Train_Data/Hotspot_prediction_using_CAD/DataSet_1_Size_1024x1024'
        elif dataset =='samsung_CAD_BE_1024_margin':
            # return 'E:/iccad-official/iccad1/gt'
            #return 'E:/3_interdata/1009+1111+0927'
            # return 'E:/iccad-official/iccad1/iccad/unsupervised'
            # return 'E:/cadsemv3'
            #return 'E:/samsung_contour/learning/98_new'
            return 'E:/5_weaklycontour/0_data/1_SamsungContour'

        elif dataset =='samsung_CAD_BE_crop_800':
            return '/data/1_data/Samsung_Image_Data/images/blind_image_data_one_NEW_VOC_style_800x800'
        elif dataset =='samsung_SEM':
            return '/data/1_data/Samsung_Image_Data/images/all_case6/ss_CAD_SEM_and_Location_VOC_style_multiClass/'
       
        elif dataset =='samsung_SEM_BE':
            return '/data/1_data/Samsung_Image_Data/images/all_case6/ss_CAD_SEM_and_Location_VOC_style_BE/'
        elif dataset =='samsung_SEM_BE_crop_100':
            return '/data/1_data/Samsung_Image_Data/images/clean_image_data_VOC_style_100x100/'
        elif dataset =='samsung_SEM_BE_crop_300':
            return '/data/1_data/Samsung_Image_Data/images/clean_image_data_VOC_style_300x300/'
        elif dataset =='samsung_SEM_BE_crop_512':
            return '/data/1_data/Samsung_Image_Data/images/clean_image_data_VOC_style_512x512/'
        elif dataset =='samsung_SEM_BE_crop_300_bt_case1':
            return '/data/1_data/Samsung_Image_Data/images/blind_image_data_one_VOC_style_300x300'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
