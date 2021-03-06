from tools.create_data import nuscenes_data_prep



if __name__ == "__main__":
    # root =  'E:/Datasets/NuScenes/new_test_run/v1.0-mini'
    # version =  'v1.0-mini'

    # PC
    # root = 'F:/Datasets/nuscenes'
    # version = 'v1.0-trainval'

    # Laptop
    # root = '/media/furqan/Terabyte/Lab/datasets/Nuscenes/v1.0-mini'
    # version = 'v1.0-mini'

    # 8 gpu server
    root ='/usb/ssd512/nusc/data'
    version = 'v1.0-trainval'

    nuscenes_data_prep(root, version,nsweeps=0, rate=0.15,filter_zero=True,create_infos=True,includeSeg=True)
    # nuscenes_data_prep('./data/nuscenes', 'v1.0-trainval', rate=0.01)
    print("Testing-finished")