# Models

Here, we provide model weights and config files for the main models from our [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/de_Geus_Task-aligned_Part-aware_Panoptic_Segmentation_through_Joint_Object-Part_Representations_CVPR_2024_paper.html).

## Pascal-PP
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pre-training</th>
<th valign="bottom">PartPQ_Pt</th>
<th valign="bottom">PartPQ_NoPt</th>
<th valign="bottom">PartPQ_All</th>
<th valign="bottom">PartSQ_Pt</th>
<th valign="bottom">PQ</th>
<th valign="bottom">config</th>
<th valign="bottom">model</th>
<!-- TABLE BODY -->
<tr><td align="left">TAPPS</td>
<td align="center">R50</td>
<td align="center">ImageNet1K</td>
<td align="center">59.6</td>
<td align="center">39.4</td>
<td align="center">44.6</td>
<td align="center">74.3</td>
<td align="center">47.1</td>
<td align="center"><a href="configs/pascal/pps/tapps_pascal_r50_in1kinit.yaml">config</a>
<td align="center"><a href="https://huggingface.co/ddegeus/TAPPS/blob/main/tapps_pascal_r50_in1kinit.bin">model</a></td>
</tr>
<tr><td align="left">TAPPS</td>
<td align="center">R50</td>
<td align="center">COCO</td>
<td align="center">67.2</td>
<td align="center">50.4</td>
<td align="center">54.7</td>
<td align="center">75.1</td>
<td align="center">57.7</td>
<td align="center"><a href="configs/pascal/pps/tapps_pascal_r50_cocoinit.yaml">config</a>
<td align="center"><a href="https://huggingface.co/ddegeus/TAPPS/blob/main/tapps_pascal_r50_cocoinit.bin">model</a></td>
</tr>
<tr><td align="left">TAPPS</td>
<td align="center">Swin-B</td>
<td align="center">COCO</td>
<td align="center">72.2</td>
<td align="center">56.3</td>
<td align="center">60.4</td>
<td align="center">78.1</td>
<td align="center">63.0</td>
<td align="center"><a href="configs/pascal/pps/tapps_pascal_swinb_cocoinit.yaml">config</a>
<td align="center"><a href="https://huggingface.co/ddegeus/TAPPS/blob/main/tapps_pascal_swinb_cocoinit.bin">model</a></td>
</tr>
</tbody></table>

## Pascal-PP-107
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pre-training</th>
<th valign="bottom">PartPQ_Pt</th>
<th valign="bottom">PartPQ_NoPt</th>
<th valign="bottom">PartPQ_All</th>
<th valign="bottom">PartSQ_Pt</th>
<th valign="bottom">PQ</th>
<th valign="bottom">config</th>
<th valign="bottom">model</th>
<!-- TABLE BODY -->
<tr><td align="left">TAPPS</td>
<td align="center">R50</td>
<td align="center">ImageNet1K</td>
<td align="center">49.8</td>
<td align="center">39.3</td>
<td align="center">42.0</td>
<td align="center">61.7</td>
<td align="center">47.2</td>
<td align="center"><a href="configs/pascal/pps/pascal_107/tapps_pascal107_r50_in1kinit.yaml">config</a>
<td align="center"><a href="https://huggingface.co/ddegeus/TAPPS/blob/main/tapps_pascal107_r50_in1kinit.bin">model</a></td>
</tr>
<tr><td align="left">TAPPS</td>
<td align="center">R50</td>
<td align="center">COCO</td>
<td align="center">55.4</td>
<td align="center">50.0</td>
<td align="center">51.4</td>
<td align="center">62.1</td>
<td align="center">57.4</td>
<td align="center"><a href="configs/pascal/pps/pascal_107/tapps_pascal107_r50_cocoinit.yaml">config</a>
<td align="center"><a href="https://huggingface.co/ddegeus/TAPPS/blob/main/tapps_pascal107_r50_cocoinit.bin">model</a></td>
</tr>
</tbody></table>

## Cityscapes-PP

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pre-training</th>
<th valign="bottom">PartPQ_Pt</th>
<th valign="bottom">PartPQ_NoPt</th>
<th valign="bottom">PartPQ_All</th>
<th valign="bottom">PartSQ_Pt</th>
<th valign="bottom">PQ</th>
<th valign="bottom">config</th>
<th valign="bottom">model</th>
<!-- TABLE BODY -->
<tr><td align="left">TAPPS</td>
<td align="center">R50</td>
<td align="center">ImageNet1K</td>
<td align="center">48.7</td>
<td align="center">63.1</td>
<td align="center">59.3</td>
<td align="center">66.8</td>
<td align="center">62.4</td>
<td align="center"><a href="configs/cityscapes/pps/tapps_cityscapes_r50_in1kinit.yaml">config</a>
<td align="center"><a href="https://huggingface.co/ddegeus/TAPPS/blob/main/tapps_cityscapes_r50_in1kinit.bin">model</a></td>
</tr>
<tr><td align="left">TAPPS</td>
<td align="center">R50</td>
<td align="center">COCO</td>
<td align="center">48.9</td>
<td align="center">65.7</td>
<td align="center">61.3</td>
<td align="center">66.9</td>
<td align="center">64.4</td>
<td align="center"><a href="configs/cityscapes/pps/tapps_cityscapes_r50_cocoinit.yaml">config</a>
<td align="center"><a href="https://huggingface.co/ddegeus/TAPPS/blob/main/tapps_cityscapes_r50_cocoinit.bin">model</a></td>
</tr>
<tr><td align="left">TAPPS</td>
<td align="center">Swin-B</td>
<td align="center">COCO</td>
<td align="center">53.0</td>
<td align="center">69.0</td>
<td align="center">64.8</td>
<td align="center">68.0</td>
<td align="center">68.0</td>
<td align="center"><a href="configs/cityscapes/pps/tapps_cityscapes_swinb_cocoinit.yaml">config</a>
<td align="center"><a href="https://huggingface.co/ddegeus/TAPPS/blob/main/tapps_cityscapes_swinb_cocoinit.bin">model</a></td>
</tr>
</tbody></table>

