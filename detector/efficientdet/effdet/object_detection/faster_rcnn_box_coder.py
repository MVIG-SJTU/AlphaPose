# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Faster RCNN box coder.

Faster RCNN box coder follows the coding schema described below:
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  th = log(h / ha)
  tw = log(w / wa)
  where x, y, w, h denote the box's center coordinates, width and height
  respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
  center, width and height respectively.

  See http://arxiv.org/abs/1506.01497 for details.
"""

import torch

from . import box_coder
from . import box_list

EPS = 1e-8


class FasterRcnnBoxCoder(box_coder.BoxCoder):
    """Faster RCNN box coder."""

    def __init__(self, scale_factors=None):
        """Constructor for FasterRcnnBoxCoder.

        Args:
            scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.
                If set to None, does not perform scaling. For Faster RCNN,
                the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].
        """
        if scale_factors:
            assert len(scale_factors) == 4
            for scalar in scale_factors:
                assert scalar > 0
        self._scale_factors = scale_factors

    @property
    def code_size(self):
        return 4

    def _encode(self, boxes, anchors):
        """Encode a box collection with respect to anchor collection.

        Args:
            boxes: BoxList holding N boxes to be encoded.
            anchors: BoxList of anchors.

        Returns:
            a tensor representing N anchor-encoded boxes of the format [ty, tx, th, tw].
        """
        # Convert anchors to the center coordinate representation.
        ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
        ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
        # Avoid NaN in division and log below.
        ha += EPS
        wa += EPS
        h += EPS
        w += EPS

        tx = (xcenter - xcenter_a) / wa
        ty = (ycenter - ycenter_a) / ha
        tw = torch.log(w / wa)
        th = torch.log(h / ha)
        # Scales location targets as used in paper for joint training.
        if self._scale_factors:
            ty *= self._scale_factors[0]
            tx *= self._scale_factors[1]
            th *= self._scale_factors[2]
            tw *= self._scale_factors[3]
        return torch.stack([ty, tx, th, tw]).T

    def _decode(self, rel_codes, anchors):
        """Decode relative codes to boxes.

        Args:
            rel_codes: a tensor representing N anchor-encoded boxes.
            anchors: BoxList of anchors.

        Returns:
            boxes: BoxList holding N bounding boxes.
        """
        ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

        ty, tx, th, tw = rel_codes.T.unbind()
        if self._scale_factors:
            ty /= self._scale_factors[0]
            tx /= self._scale_factors[1]
            th /= self._scale_factors[2]
            tw /= self._scale_factors[3]
        w = torch.exp(tw) * wa
        h = torch.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2.
        return box_list.BoxList(torch.stack([ymin, xmin, ymax, xmax]).T)
