import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.ops import sigmoid_focal_loss

from util import box_ops
from util.misc import accuracy, nested_tensor_from_tensor_list, interpolate


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if output_last_feature:
            return h4
        else:
            return [h1, h2, h3, h4]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes,weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.softmax1=torch.nn.Softmax()
        self.softmax2=torch.nn.Softmax()

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            #             'labels': self.loss_labels,
            #             'cardinality': self.loss_cardinality,
            #             'boxes': self.loss_boxes,
            #             'masks': self.loss_masks
            'content': self.loss_content,
            'style': self.loss_style

        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def calc_mean_std(self, features):
        """

        :param features: shape of features -> [batch_size, c, h, w]
        :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
        """

        batch_size, c = features.size()[:2]
        features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
        features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
        return features_mean, features_std

    def loss_content(self, out_features, t):

        loss = 0
        for out_i, target_i in zip(out_features, t):
            #             print("out_i.shape,target_i.shape:",out_i.shape,target_i.shape)
            loss += F.mse_loss(out_i, target_i)
        return loss

    def loss_content_last(self, out_features, t):
        return F.mse_loss(out_features, t)

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)

        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def tv_loss(self, img):
        """
        Compute total variation loss.
        Inputs:
        - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
        - tv_weight: Scalar giving the weight w_t to use for the TV loss.
        Returns:
        - loss: PyTorch Variable holding a scalar giving the total variation loss
          for img weighted by tv_weight.
        """
        # Your implementation should be vectorized and not require any loops!
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N, C, H, W = img.shape
        x1 = img[:, :, 0:H - 1, :]
        x2 = img[:, :, 1:H, :]
        y1 = img[:, :, :, 0:W - 1]
        y2 = img[:, :, :, 1:W]
        loss = ((x2 - x1).pow(2).sum() + (y2 - y1).pow(2).sum())
        return loss

    def loss_style_gram(self, output_middle_features, style_middle_features):
        target_gram = self.gram_matrix(style_middle_features)
        output_gram = self.gram_matrix(output_middle_features)
        return F.mse_loss(output_gram, target_gram)

    def loss_style_gram_multiple(self, content_middle_features, style_middle_features):
        loss = 0
        #         print("content_middle_features.shape, style_middle_features.shape:",content_middle_features.shape, style_middle_features.shape)
        for c, s in zip(content_middle_features, style_middle_features):
            target_gram = self.gram_matrix(c)
            output_gram = self.gram_matrix(s)
            loss += F.mse_loss(output_gram, target_gram)
        return loss

    def loss_style_adain(self, content_middle_features, style_middle_features):
        loss = 0
        #         print("content_middle_features.shape, style_middle_features.shape:",content_middle_features.shape, style_middle_features.shape)
        for c, s in zip(content_middle_features, style_middle_features):
            #             print("c.shape,s.shape:",c.shape,s.shape)
            c_mean, c_std = self.calc_mean_std(c)
            s_mean, s_std = self.calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss

    def forward_general(self, outputs, targets_content, targets_style):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output_middle_features = self.vgg_encoder(outputs, output_last_feature=False)
        content_middle_features = self.vgg_encoder(targets_content.tensors, output_last_feature=False)
        loss_c = self.loss_content(output_middle_features, content_middle_features)
        # gram loss:
        output_features = self.vgg_encoder(outputs, output_last_feature=True)
        style_features = self.vgg_encoder(targets_style.tensors, output_last_feature=True)
        loss_s = self.loss_style_gram(output_features, style_features)
        #         loss_s = self.loss_style(output_features,style_features)+ self.loss_style(style_res_features,style_features)

        losses = {
            'loss_content': loss_c,
            'loss_style': loss_s

        }
        return losses

    def forward_adain(self, outputs, targets_content, targets_style):  # _adain
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        content_features = self.vgg_encoder(targets_content.tensors, output_last_feature=True)
        output_features = self.vgg_encoder(outputs, output_last_feature=True)

        loss_c = self.loss_content_last(output_features, content_features)

        # adain loss:
        style_middle_features = self.vgg_encoder(targets_style.tensors, output_last_feature=False)
        output_middle_features = self.vgg_encoder(outputs, output_last_feature=False)
        loss_s = self.loss_style_adain(output_middle_features, style_middle_features)

        losses = {
            'loss_content': loss_c,
            'loss_style': loss_s

        }
        return losses

    def forward(self, outputs, targets_content, targets_style):  # _hybrid
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        #         print("outputs.shape, targets_content.shape,targets_style.shape:",outputs.shape, targets_content.shape,targets_style.shape)

        #         output_middle_features = self.vgg_encoder(outputs, output_last_feature=False)
        #         content_middle_features = self.vgg_encoder(targets_content.tensors, output_last_feature=False)
        #         loss_c = self.loss_content(output_middle_features, content_middle_features)

        content_features = self.vgg_encoder(targets_content, output_last_feature=True)
        output_features = self.vgg_encoder(outputs, output_last_feature=True)
        loss_c = self.loss_content_last(output_features, content_features)

        # adain loss:
        style_middle_features = self.vgg_encoder(targets_style, output_last_feature=False)
        output_middle_features = self.vgg_encoder(outputs, output_last_feature=False)
        loss_s = self.loss_style_adain(output_middle_features, style_middle_features)

        #         loss_s = self.loss_style_gram_multiple(output_middle_features, style_middle_features)

        loss_tv = self.tv_loss(outputs)

        losses = {
            'loss_content': loss_c,
            'loss_style': loss_s,
            'loss_tv': loss_tv

        }
        return losses
def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes