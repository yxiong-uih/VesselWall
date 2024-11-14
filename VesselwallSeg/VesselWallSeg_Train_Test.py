from VesselWallSeg.seg_base_train import SegBaseTrain as Train
from VesselWallSeg.seg_base_test import SegBaseTest as Test


def SegBaseTrain2D_UT():

    config_file = r'F:/MRPlaque/VesselWallSegmentation/11/PlaqueSeg_Train.py'
    T = Train(config_file)
    T.train_model()    #训练形状模型


def SegBaseTest2D_UT():


    net_file = r'F:/MRPlaque/VesselWallSegmentation/Result/InceptionUnet/params.pth' ##20  170
    gpu_ids = 0
    save_results_path=r'F:/MRPlaque/VesselWallSegmentation/Result/InceptionUnet/'
    net_id= 0
    T = Test(net_file,gpu_ids,save_results_path,net_id)
    test_file_list = r'F:/MRPlaque/ResultforPaper/result0404/InceptionUnet/TestData1.txt'
  
    T.test(test_file_list=test_file_list,is_gt=True,is_heatmap=True)
    
    
    
if __name__ == '__main__':
   #SegBaseTrain2D_UT()
   SegBaseTest2D_UT()
   #SegBaseTrain2D_UT()

