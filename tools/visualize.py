import numpy as np
# import open3d
import cv2
import matplotlib.pyplot as plt
import yaml


def visualize_pcloud(scan_points,scan_labels):
    # vis = open3d.visualization.Visualizer()
    # vis.create_window()
    label_colormap = yaml.safe_load(open('configs/nusc/colormap.yaml', 'r'))
    label_colormap =label_colormap['short_color_map']
    # rendering the pcloud in open3d
    pcd = open3d.geometry.PointCloud()
    # scan_points = scan_points.numpy()
    scan_points = scan_points[:,:3]
    pcd.points = open3d.utility.Vector3dVector(scan_points)
    # scan_labels = scan_labels.numpy()
    # scan_labels = scan_labels[scan_labels != -1]
    scan_labels = scan_labels.astype(np.int)
    colors = np.array([label_colormap[x] for x in scan_labels])
    pcd.colors = open3d.utility.Vector3dVector(colors / 255.0)
    vis = open3d.visualization.VisualizerWithKeyCallback()
    # vis.create_window(width=width, height=height, left=100)
    # vis.add_geometry(pcd)
    vis = open3d.visualization.draw_geometries([pcd])
    open3d.visualization.ViewControl()


def visualize_multi_pcloud(scan_points,scan_labels):
    # vis = open3d.visualization.Visualizer()
    # vis.create_window()
    label_colormap = yaml.safe_load(open('configs/nusc/colormap.yaml', 'r'))
    label_colormap = label_colormap['short_color_map']
    # rendering the pcloud in open3d
    pcd = open3d.geometry.PointCloud()
    # scan_points = scan_points.numpy()
    scan_points = scan_points[:, :3]
    pcd.points = open3d.utility.Vector3dVector(scan_points)
    # scan_labels = scan_labels.numpy()
    # scan_labels = scan_labels[scan_labels != -1]
    scan_labels = scan_labels.astype(np.int)
    colors = np.array([label_colormap[x] for x in scan_labels])
    pcd.colors = open3d.utility.Vector3dVector(colors / 255.0)
    vis = open3d.visualization.VisualizerWithKeyCallback()
    # vis.create_window(width=width, height=height, left=100)
    # vis.add_geometry(pcd)
    vis = open3d.visualization.draw_geometries([pcd])
    open3d.visualization.ViewControl()

    from open3d.visualization import Visualizer
    vis = Visualizer()
    vis.create_window()



def visualize_camera(img):
    img= np.concatenate(img)
    cv2.imwrite('data.jpg', img)
    cv2.imshow('img_show',img)
    cv2.waitKey(0)


def visualize_2dlabels(img):
    label_colormap = yaml.safe_load(open('configs/nusc/colormap.yaml', 'r'))
    label_colormap = label_colormap['short_color_map']
    colors = np.array([[label_colormap[val] for val in row] for row in img], dtype='B')
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
    cv2.imwrite('range_projection.jpg', colors)
    cv2.imshow('img_show', colors)
    cv2.waitKey(0)


def plot_colormap(original=True):

    general_to_label = {0: 'ignore', 1: 'ignore', 2: 'pedestrian', 3: 'pedestrian',
                        4: 'pedestrian', 5: 'ignore', 6: 'pedestrian', 7: 'ignore',
                        8: 'ignore', 9: 'barrier', 10: 'ignore', 11: 'ignore',
                        12: 'traffic_cone', 13: 'ignore', 14: 'bicycle', 15: 'bus',
                        16: 'bus', 17: 'car', 18: 'construction_vehicle', 19: 'ignore',
                        20: 'ignore', 21: 'motorcycle', 22: 'trailer', 23: 'truck',
                        24: 'road', 25: 'ignore', 26: 'sidewalk', 27: 'terrain',
                        28: 'building', 29: 'ignore', 30: 'vegetation', 31: 'ignore'}

    mapped = { 'ignore': 0,'car': 1,'pedestrian': 2,  'bicycle': 3,'motorcycle': 4,
          'bus': 5, 'truck': 6, 'construction_vehicle': 7,'trailer': 8, 'barrier': 9,
           'traffic_cone': 10,'driveable_surface': 11, 'other_flat': 12,'sidewalk': 13,
          'terrain': 14, 'manmade': 15,'vegetation': 16}

    if original :
        # label_to_general = dict((y,x) for x,y in general_to_label.items())
        label_colormap = yaml.safe_load(open('configs/nusc/colormap.yaml', 'r'))
        label_colormap = label_colormap['color_map']

        for ind,(key,val) in enumerate(general_to_label.items()):
            plt.subplot(6,6,ind+1)
            arr = np.full(25,np.int(key),dtype=np.int32)
            colors = np.array([label_colormap[x] for x in arr])
            colors = colors.reshape(5,5,-1)
            plt.subplots_adjust(wspace=0.4,hspace=0.8)
            plt.title(val,fontsize=8)
            plt.axis('off')
            plt.imshow(colors)
    else:
        label_colormap = yaml.safe_load(open('configs/nusc/colormap.yaml', 'r'))
        label_colormap = label_colormap['short_color_map']

        for ind, (key, val) in enumerate(mapped.items()):
            plt.subplot(4, 5, ind + 1)
            arr = np.full(25, np.int(val), dtype=np.int32)
            colors = np.array([label_colormap[x] for x in arr])
            colors = colors.reshape(5, 5, -1)
            plt.subplots_adjust(wspace=0.4, hspace=0.8)
            plt.title(key, fontsize=8)
            plt.axis('off')
            plt.imshow(colors)

    plt.show()


data= np.load('augimg0.npy')
print(data.shape)