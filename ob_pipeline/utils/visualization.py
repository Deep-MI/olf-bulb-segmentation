import sys
sys.path.append('../')
sys.path.append('/../../')
sys.path.append('../../../')
import numpy as np
import os
#os.environ['MPLCONFIGDIR'] = "./tmp"
#nilearn-0.7.1


def get_cmap():
    import matplotlib.colors
    colors=[[0,0,1],[1,0,0]]

    cmap = matplotlib.colors.ListedColormap(colors)

    return cmap


def plot_coronal_predictions(images_batch=None,pred_batch=None,img_per_row=4):
    from skimage import color
    import matplotlib.pyplot as plt
    import torch
    from torchvision import  utils
    plt.ioff()
    DEFAULT_COLORS= [(0,0,255),(255,0,0)] # 1: blue 2 : red

    FIGSIZE = 2
    FIGDPI = 100

    ncols=2

    nrows=1

    fig, ax = plt.subplots(nrows,ncols)

    grid_size=(images_batch.shape[0]/img_per_row,img_per_row)

    # adjust layout
    fig.set_size_inches([FIGSIZE * ncols * grid_size[1] , FIGSIZE * nrows * grid_size[0]])
    fig.set_dpi(FIGDPI)
    fig.set_facecolor('black')
    fig.set_tight_layout({'pad': 0})
    fig.subplots_adjust(wspace=0,hspace=0)

    pos=0


    images = torch.from_numpy(images_batch.copy())
    images = torch.unsqueeze(images,1)
    grid = utils.make_grid(images.cpu(), nrow=img_per_row,normalize=True)
    #ax[pos].imshow(grid.numpy().transpose(1, 2, 0), cmap='gray',origin='lower')
    ax[pos].imshow(np.fliplr(grid.numpy().transpose(1, 2, 0)), cmap='gray', origin='lower')
    ax[pos].set_axis_off()
    ax[pos].set_aspect('equal')
    ax[pos].margins(0, 0)
    ax[pos].set_title('T2 input image (1 to N)',color='white')
    pos += 1

    pred=torch.from_numpy(pred_batch.copy())
    pred = torch.unsqueeze(pred, 1)
    pred_grid = utils.make_grid(pred.cpu(), nrow=img_per_row)[0] #dont take the channels axis from grid
    #pred_grid=color.label2rgb(pred_grid.numpy(),grid.numpy().transpose(1, 2, 0),alpha=0.6,bg_label=0,colors=DEFAULT_COLORS)
    pred_grid = color.label2rgb(pred_grid.numpy(), grid.numpy().transpose(1, 2, 0), alpha=0.6, bg_label=0,bg_color=None,
                                colors=DEFAULT_COLORS)
    #ax[pos].imshow(pred_grid,origin='lower')
    ax[pos].imshow(np.fliplr(pred_grid), origin='lower')
    ax[pos].set_axis_off()
    ax[pos].set_aspect('equal')
    ax[pos].margins(0, 0)
    ax[pos].set_title('Predictions (1 to N). Left OB (blue); Right OB (Red)',color='white')
    ax[pos].margins(0, 0)

    return fig



def plot_qc_images(save_dir,image,prediction,padd=30):
    from scipy import ndimage
    from nilearn import plotting
    from utils import image_utils

    qc_dir=os.path.join(save_dir,'QC')


    plotting.plot_roi(bg_img=image, roi_img=prediction,
                      display_mode='ortho', output_file=os.path.join(qc_dir,'overall_screenshot.png'), draw_cross=False,
                      cmap=get_cmap())

    plane = 'coronal'
    mod_image = image_utils.plane_swap(image.get_fdata(), plane)
    mod_pred = image_utils.plane_swap(prediction.get_fdata(), plane)

    idx = np.where(mod_pred > 0)
    idx = np.unique(idx[0])


    if len(idx) > 0:

        crop_image = mod_image[np.min(idx) - 4:np.max(idx) + 4, :, :]

        crop_seg = mod_pred[np.min(idx) - 4:np.max(idx) + 4, :, :]

        cm = ndimage.measurements.center_of_mass(crop_seg > 0)

        cm = np.array(cm).astype(int)

        crop_image = crop_image[:, cm[1] - padd:cm[1] + padd, cm[2] - padd:cm[2] + padd]
        crop_seg = crop_seg[:, cm[1] - padd:cm[1] + padd, cm[2] - padd:cm[2] + padd]

    else:
        depth = mod_image.shape[0] // 2
        crop_image = mod_image[depth - 8:depth + 8, :, :]
        crop_seg = mod_pred[depth - 8:depth + 8, :, :]

        cm = [crop_image.shape[0]//2,crop_image.shape[1]//2,crop_image.shape[2]//2]
        cm = np.array(cm).astype(int)


        crop_image = crop_image[:, cm[1] - padd:cm[1] + padd, cm[2] - padd:cm[2] + padd]
        crop_seg = crop_seg[:, cm[1] - padd:cm[1] + padd, cm[2] - padd:cm[2] + padd]


    fig = plot_coronal_predictions(crop_image, crop_seg, img_per_row=4)

    fig.savefig(os.path.join(qc_dir,'{}_screenshot.png'.format(plane)), dpi=100, transparent=False)


