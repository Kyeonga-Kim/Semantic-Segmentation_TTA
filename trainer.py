import torch
from torch import nn
import torch.nn.functional as F
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
import models
from collections import OrderedDict
import json
import ttach as tta
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import csv
import PIL




class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)
        
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        if self.device ==  torch.device('cpu'): prefetch = False
        if prefetch:
            #self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True


    def _valid_epoch(self, epoch):
        snapshot_path = './result'
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        header = ['Epoch',  'Val Loss', 'pixAcc', 'mIoU', 'ClassIoU']

        if not os.path.isfile(snapshot_path + '/log.csv'):
            with open(snapshot_path + '/log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)        

        if not os.path.exists('outputs_2'):
            os.makedirs('outputs_2')

        #tensorboard
        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        writer_dir = os.path.join('./logs/', 'TTA' ,start_time)
        #self.writer = SummaryWriter()
        self.writer = SummaryWriter(writer_dir) 

        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        # Model
        with open('./config.json', 'r') as f:
            config = json.load(f)

        availble_gpus = list(range(torch.cuda.device_count()))
        device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
 
    
        #model definition
        # num_classes = self.val_loader.dataset.num_classes
        palette =  self.val_loader.dataset.palette
        #model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])

        # Load checkpoint
        # checkpoint = torch.load('/home/kka0602/pytorch-segmentation/pretrained/PSPnet.pth', map_location=device)
        # if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        #     checkpoint = checkpoint['state_dict']
        # # If during training, we used data parallel
        # if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        #     # for gpu inference, use data parallel
        #     if "cuda" in device.type:
        #         model = torch.nn.DataParallel(model)
        #     else:
        #     # for cpu inference, remove module
        #         new_state_dict = OrderedDict()
        #         for k, v in checkpoint.items():
        #             name = k[7:]
        #             new_state_dict[name] = v
        #         checkpoint = new_state_dict
        # # load
        # self.model.load_state_dict(checkpoint)

        self.model.to(device)
        self.model.eval()

        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                #data, target = data.to(self.device), target.to(self.device)
                
                #output_base = self.model(data)['out'] #['out', 'aux']
          
                #TTA
                self.tta_model = tta.SegmentationTTAWrapper(self.model, tta.aliases.d4_transform(), merge_mode='mean', output_mask_key='out')
                output = self.tta_model(data)['out'] #tta ouput #dict_key['out]

                #LOSS
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                # if len(val_visual) < 15:
                #     target_np = target.data.cpu().numpy()
                #     output_np = output.data.max(1)[1].cpu().numpy()
                #     val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, classIOU = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Loss: {:.4f}, PixelAcc: {:.4f}, Mean IoU: {:.4f}|'.format( epoch,
                                                self.total_loss.average,
                                                pixAcc, mIoU))

  

                # Saved Visualization
                # target_np = target.squeeze(0).cpu().numpy() #target #(1, 480, 480)
                # target_np = colorize_mask(target_np, palette)
                # #target_np = PIL.Image.fromarray(target_np.astype(np.uint8)).convert('P')
             

                # prediction_base = output_base.squeeze(0).cpu().numpy()
                # prediction_base = F.softmax(torch.from_numpy(prediction_base), dim=0).argmax(0).cpu().numpy()
                # colorized_mask_base = colorize_mask(prediction_base, palette) #Image
 
                # prediction = output.squeeze(0).cpu().numpy()
                # prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
                # colorized_mask = colorize_mask(prediction, palette)

                # concat_colorized_mask = Image.new("RGB",(480, 480))

                # concat_colorized_mask.paste(im=target_np, box=(0, 0))
                # concat_colorized_mask.paste(im=colorized_mask_base, box=(480, 0))
                #concat_colorized_mask.paste(im=colorized_mask, box=(0, 0))
                # #save
                #concat_colorized_mask.save(os.path.join('outputs_2/', str(batch_idx)+'.png'))


            # # WRTING & VISUALIZING THE MASKS
            # val_img = []
            # # palette = self.train_loader.dataset.palette
            # for d, t, o in val_visual:
            #     d = self.restore_transform(d)
            #     t, o = colorize_mask(t, palette), colorize_mask(o, palette)
            #     d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
            #     [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
            #     val_img.extend([d, t, o])
            # val_img = torch.stack(val_img, 0)
            # val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            # self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

            result = [epoch, self.total_loss.average, pixAcc, mIoU, classIOU]

            with open('result' + '/log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)  

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
            
        }
       
    def save_images(self, image, mask, output_path, image_file, palette):
        # Saves the image, the model output and the results after the post processing
        w, h = image.size
        image_file = os.path.basename(image_file).split('.')[0]
        colorized_mask = colorize_mask(mask, palette)
        colorized_mask.save(os.path.join(output_path, image_file+'.png'))
