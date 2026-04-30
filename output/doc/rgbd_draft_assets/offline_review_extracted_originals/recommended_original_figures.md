# 论文附图优先筛选清单（原图，不改图）

这些不是重新生成的图，而是 `offline_brick_recognition_review.py` 的原始输出图，只是按论文用途筛了一遍。

## A. 正常一致案例：case_0005
用途：展示系统能完整记录主抓取、二次定位、RGB 复盘和深度几何复盘。
- 主抓取 / RGB复盘总图：`case_0005_primary_pick_analysis.png`
- 主抓取 / 深度几何总图：`case_0005_primary_pick_depth_analysis.png`
- 主抓取 / 深度高度图：`case_0005_primary_pick_depth_height_map.png`
- 主抓取 / 深度连通区域：`case_0005_primary_pick_depth_regions.png`
- 主抓取 / 深度矩形拟合：`case_0005_primary_pick_depth_rectangles.png`
- 二次定位 / RGB复盘总图：`case_0005_secondary_alignment_analysis.png`
- 二次定位 / 深度几何总图：`case_0005_secondary_alignment_depth_analysis.png`
- 二次定位 / 深度高度图：`case_0005_secondary_alignment_depth_height_map.png`
- 二次定位 / 深度连通区域：`case_0005_secondary_alignment_depth_regions.png`
- 二次定位 / 深度矩形拟合：`case_0005_secondary_alignment_depth_rectangles.png`

## B. 明显偏差案例：case_0006 secondary_alignment
用途：展示 review_needed，说明离线复盘能暴露运行态与离线候选偏差。
- 二次定位 / RGB复盘总图：`case_0006_secondary_alignment_analysis.png`
- 二次定位 / 深度几何总图：`case_0006_secondary_alignment_depth_analysis.png`
- 二次定位 / 深度高度图：`case_0006_secondary_alignment_depth_height_map.png`
- 二次定位 / 深度连通区域：`case_0006_secondary_alignment_depth_regions.png`
- 二次定位 / 深度矩形拟合：`case_0006_secondary_alignment_depth_rectangles.png`

## C. 离线缺失案例：case_0002 secondary_alignment
用途：展示 offline_missing，说明异常样本筛查和失败归因。
- 二次定位 / RGB复盘总图：`case_0002_secondary_alignment_analysis.png`
- 二次定位 / 深度几何总图：`case_0002_secondary_alignment_depth_analysis.png`
- 二次定位 / 深度高度图：`case_0002_secondary_alignment_depth_height_map.png`
- 二次定位 / 深度连通区域：`case_0002_secondary_alignment_depth_regions.png`
- 二次定位 / 深度矩形拟合：`case_0002_secondary_alignment_depth_rectangles.png`

## D. 小偏差复核案例：case_0001 secondary_alignment
用途：展示二次定位阶段较小偏差样本，可和 case_0006 对比。
- 二次定位 / RGB复盘总图：`case_0001_secondary_alignment_analysis.png`
- 二次定位 / 深度几何总图：`case_0001_secondary_alignment_depth_analysis.png`
- 二次定位 / 深度高度图：`case_0001_secondary_alignment_depth_height_map.png`
- 二次定位 / 深度连通区域：`case_0001_secondary_alignment_depth_regions.png`
- 二次定位 / 深度矩形拟合：`case_0001_secondary_alignment_depth_rectangles.png`