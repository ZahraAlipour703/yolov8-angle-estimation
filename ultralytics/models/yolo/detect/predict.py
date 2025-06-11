# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
import numpy as np
import cv2


class DetectionPredictor(BasePredictor):
    """
    Extends BasePredictor for detection + optional keypoint‐angle rendering.
    """

    def postprocess(self, preds, img, orig_imgs):
        save_feats = getattr(self, "_feats", None) is not None

        # NMS (optionally returning indices for feature extraction)
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        # ensure orig_imgs is a list of numpy arrays
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # if we're saving features, split preds into boxes & idxs
        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        # build Results
        results = self.construct_results(preds, img, orig_imgs)

        # optional: draw angles between triplets of keypoints
        if getattr(self.args, "draw_keypoint_angles", False):
            triplets = getattr(self.args, "keypoint_angle_triplets", [])
            for r in results:
                if r.keypoints is not None:
                    r.orig_img = self.draw_keypoint_angles(
                        r.orig_img,
                        r.keypoints.data[0].cpu().numpy().tolist(),
                        triplets,
                    )

        # attach features if present
        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f

        return results

    def get_obj_feats(self, feat_maps, idxs):
        """Extract per‐object features from feature maps."""
        import torch

        # spatial pooling size
        s = min(f.shape[1] for f in feat_maps)
        pooled = []
        for f in feat_maps:
            B, C, H, W = f.shape
            # reshape to (B, N_anchors, s, C//s) then average over the spatial dim
            m = (
                f.permute(0, 2, 3, 1)
                 .reshape(B, -1, s, C // s)
                 .mean(dim=-1)
            )
            pooled.append(m)
        obj_feats = torch.cat(pooled, dim=1)  # (B, total_anchors, C//s)
        # for each image, select the features for kept indices
        return [feats[i] if len(i) else [] for feats, i in zip(obj_feats, idxs)]

    def construct_results(self, preds, img, orig_imgs):
        """Turn raw predictions + images into a list of Results."""
        return [
            self.construct_result(p, img, orig_img, path)
            for p, orig_img, path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """Scale box coords back to original image and wrap into Results."""
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        # columns: x,y,w,h,conf,class
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])

    def draw_keypoint_angles(self, image, keypoints, pairs):
        """
        Annotate `image` with the angle at keypoint B for each (A, B, C) triplet.
        """
        for a_i, b_i, c_i in pairs:
            if max(a_i, b_i, c_i) >= len(keypoints):
                continue
            a, b, c = keypoints[a_i], keypoints[b_i], keypoints[c_i]
            # only draw if all confidences > 0.5
            if a[2] > 0.5 and b[2] > 0.5 and c[2] > 0.5:
                a_pt = np.array(a[:2])
                b_pt = np.array(b[:2])
                c_pt = np.array(c[:2])
                ang = self.compute_angle(a_pt, b_pt, c_pt)
                cv2.putText(
                    image,
                    f"{int(ang)}°",
                    tuple(b_pt.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        return image

    @staticmethod
    def compute_angle(a, b, c):
        """
        Compute the 2D angle at point b formed by points a‐b‐c, in degrees.
        """
        ba = a - b
        bc = c - b
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
