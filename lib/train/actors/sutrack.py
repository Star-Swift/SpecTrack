from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
from lib.train.admin import multigpu
from lib.utils.heapmap_utils import generate_heatmap

class SUTrack_Actor(BaseActor):
    """ Actor for training the sutrack"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE
        # MoCE辅助损失权重
        self.moce_aux_weight = getattr(cfg.TRAIN, 'MOCE_AUX_WEIGHT', 0.01)
        # 光谱对比损失权重
        self.spectral_contrastive_weight = getattr(cfg.TRAIN, 'SPECTRAL_CONTRASTIVE_WEIGHT', 0.0)

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        b = data['search_images'].shape[1]   # n,b,c,h,w
        search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b,dim=0)  # (n*b, c, h, w)
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b,dim=0)
        template_anno_list = data['template_anno'].view(-1, *data['template_anno'].shape[2:]).split(b,dim=0)

        text_src = None

        # task_class
        task_index_batch = [self.cfg.MODEL.TASK_INDEX[key.upper()] for key in data['dataset']]
        # task_index_batch = range(len(data['dataset']))
        task_index_batch = torch.tensor(task_index_batch).cuda() #torch.Size([bs])

        enc_opt = self.net(template_list=template_list,
                           search_list=search_list,
                           template_anno_list=template_anno_list,
                           text_src=text_src,
                           task_index=task_index_batch,
                           mode='encoder') # forward the encoder
        outputs, task_class_output = self.net(feature=enc_opt, mode="decoder")
        # outputs = self.net(feature=enc_opt, mode="decoder")
        # task_class_output = self.net(feature=enc_opt, mode="task_decoder")
        task_class_output = task_class_output.view(-1, task_class_output.size(-1))
        outputs['task_class'] = task_class_output
        outputs['task_class_label'] = task_index_batch
        outputs['enc_opt'] = enc_opt

        return outputs

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # task classification loss
        task_cls_loss = self.objective['task_cls'](pred_dict['task_class'], pred_dict['task_class_label'])

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.ENCODER.STRIDE) # list of torch.Size([b, H, W])
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1) # torch.Size([b, 1, H, W])

        # Get boxes
        pred_boxes = pred_dict['pred_boxes'] # torch.Size([b, 1, 4])
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # compute MoCE auxiliary loss
        moce_aux_loss = torch.tensor(0.0, device=l1_loss.device)
        if self.training:
            net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
            if hasattr(net, 'get_moce_aux_loss'):
                moce_aux_loss = net.get_moce_aux_loss()
                if not isinstance(moce_aux_loss, torch.Tensor):
                    moce_aux_loss = torch.tensor(moce_aux_loss, device=l1_loss.device)

        # compute spectral contrastive loss
        spectral_loss = torch.tensor(0.0, device=l1_loss.device)
        if self.training and self.spectral_contrastive_weight > 0:
            net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
            if hasattr(net, 'compute_spectral_contrastive_loss') and 'enc_opt' in pred_dict:
                spectral_loss = net.compute_spectral_contrastive_loss(
                    pred_dict['enc_opt'],
                    gt_gaussian_maps.squeeze(1)  # [B, H, W] at patch resolution
                )

        # weighted sum
        loss = (self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss +
                self.loss_weight['task_cls'] * task_cls_loss +
                self.moce_aux_weight * moce_aux_loss +
                self.spectral_contrastive_weight * spectral_loss)

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/task_class": task_cls_loss.item(),
                      "Loss/moce_aux": moce_aux_loss.item() if isinstance(moce_aux_loss, torch.Tensor) else moce_aux_loss,
                      "Loss/spectral_contrast": spectral_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss