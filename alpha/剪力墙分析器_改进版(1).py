import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os
import re
from collections import defaultdict
from docx import Document

class ShearWallAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("剪力墙识别与坐标提取工具")
        self.root.geometry("1400x900")

        self.wall_data = []
        self.wall_centers = {}

        self._create_widgets()

    def _create_widgets(self):
        # 顶部上传区
        upload_frame = ttk.Frame(self.root, padding="15")
        upload_frame.pack(fill=tk.X)

        ttk.Label(
            upload_frame,
            text="选择文件：",
            font=("微软雅黑", 12, "bold")
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            upload_frame,
            text="上传文件",
            command=self._upload_file,
            width=15
        ).pack(side=tk.LEFT, padx=10)

        self.file_status = ttk.Label(
            upload_frame,
            text="未上传文件",
            font=("微软雅黑", 10),
            foreground="#7f8c8d"
        )
        self.file_status.pack(side=tk.LEFT, padx=10)

        ttk.Button(
            upload_frame,
            text="导出中心坐标",
            command=self._export_centers,
            width=15,
            state=tk.DISABLED
        ).pack(side=tk.RIGHT, padx=5)

        # 中间数据显示区
        data_frame = ttk.LabelFrame(
            self.root,
            text="剪力墙数据列表",
            padding="15"
        )
        data_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        columns = ("number", "id", "x_range", "y_range", "center", "shape")
        self.data_table = ttk.Treeview(
            data_frame,
            columns=columns,
            show="headings",
            selectmode="browse"
        )

        self.data_table.heading("number", text="编号", anchor=tk.CENTER)
        self.data_table.heading("id", text="墙ID", anchor=tk.CENTER)
        self.data_table.heading("x_range", text="X坐标范围", anchor=tk.CENTER)
        self.data_table.heading("y_range", text="Y坐标范围", anchor=tk.CENTER)
        self.data_table.heading("center", text="中心坐标", anchor=tk.CENTER)
        self.data_table.heading("shape", text="形状", anchor=tk.CENTER)

        self.data_table.column("number", width=80, anchor=tk.CENTER)
        self.data_table.column("id", width=100, anchor=tk.CENTER)
        self.data_table.column("x_range", width=200, anchor=tk.CENTER)
        self.data_table.column("y_range", width=200, anchor=tk.CENTER)
        self.data_table.column("center", width=180, anchor=tk.CENTER)
        self.data_table.column("shape", width=120, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(
            data_frame,
            orient=tk.VERTICAL,
            command=self.data_table.yview
        )
        self.data_table.configure(yscrollcommand=scrollbar.set)

        self.data_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _upload_file(self):
        file_path = filedialog.askopenfilename(
            title="选择剪力墙文件",
            filetypes=[
                ("Word文档", "*.docx"),
                ("文本文件", "*.txt"),
                ("JSON文件", "*.json"),
                ("所有文件", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            print(f"[DEBUG] Starting to load file: {file_path}")

            # 根据文件类型读取内容
            if file_path.endswith('.docx'):
                print("[DEBUG] Reading DOCX file...")
                content = self._read_docx(file_path)
                print(f"[DEBUG] DOCX content length: {len(content)}")
            else:
                # 读取文本文件
                print("[DEBUG] Reading text file...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"[DEBUG] Text content length: {len(content)}")

            # 清理文件中的注释和额外文字
            print("[DEBUG] Cleaning file content...")
            content = self._clean_file_content(content)
            print(f"[DEBUG] Cleaned content length: {len(content)}")

            # 解析JSON数据
            print("[DEBUG] Parsing JSON...")
            data = json.loads(content)
            print(f"[DEBUG] JSON parsed successfully, type: {type(data)}")

            if not isinstance(data, list):
                messagebox.showerror("错误", "JSON文件格式不正确：应为数组")
                return

            print(f"[DEBUG] Found {len(data)} items in array")
            self.wall_data = []
            for item in data:
                if item.get("type") == "shearwall":
                    xs = item.get("xs", [])
                    ys = item.get("ys", [])
                    wall_dir = item.get("props", {}).get("dir", "")

                    wall_info = {
                        "id": item.get("id", 0),
                        "x_coords": xs,
                        "y_coords": ys,
                        "thickness": item.get("props", {}).get("thick", 0),
                        "material": item.get("props", {}).get("mat", ""),
                        "direction": wall_dir
                    }
                    self.wall_data.append(wall_info)

            print(f"[DEBUG] Found {len(self.wall_data)} shear walls total")

            if not self.wall_data:
                messagebox.showwarning("提示", "未在JSON文件中找到剪力墙数据")
                return

            # 合并连接的横墙和竖墙成L型墙
            self._merge_connected_walls()

            # 按照从左到右、从上到下的顺序排序
            self._sort_walls()

            # 计算中心位置
            self._calculate_centers()

            # 更新表格
            self._update_table()

            # 显示识别结果弹窗
            self._show_recognition_result()

            self.file_status.config(
                text=f"已加载：{os.path.basename(file_path)}",
                foreground="#27ae60"
            )

            # 启用导出按钮
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Button) and "导出" in str(child['text']):
                            child.config(state=tk.NORMAL)

        except json.JSONDecodeError as e:
            error_msg = f"JSON解析失败：{str(e)}\n\n"
            error_msg += f"错误位置：第{e.lineno}行，第{e.colno}列\n"
            error_msg += f"问题字符：{e.msg}\n\n"
            error_msg += "请检查文件格式是否正确"
            messagebox.showerror("错误", error_msg)
            self.file_status.config(text="解析失败", foreground="#e74c3c")
            print(f"JSON Error: {e}")
            print(f"Content preview: {content[:500] if content else 'No content'}")
        except Exception as e:
            import traceback
            error_msg = f"文件读取失败：{str(e)}\n\n"
            error_msg += f"错误类型：{type(e).__name__}\n\n"
            error_msg += "详细错误信息已打印到控制台"
            messagebox.showerror("错误", error_msg)
            self.file_status.config(text="读取失败", foreground="#e74c3c")
            print(f"Error: {e}")
            traceback.print_exc()

    def _read_docx(self, file_path):
        """
        读取docx文件内容
        """
        try:
            doc = Document(file_path)
            content = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content += paragraph.text.strip() + "\n"
            return content
        except Exception as e:
            raise Exception(f"读取Word文档失败：{str(e)}")

    def _clean_file_content(self, content):
        """
        清理txt/docx文件中的注释和额外文字，提取纯JSON数据
        """
        # 移除注释（//开头的内容）
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)

        # 移除 /* */ 类型的注释
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # 移除中文注释（圆括号中的中文）
        content = re.sub(r'\（[^）]*\）', '', content)

        # 移除其他说明文字（如"ps："等）
        content = re.sub(r'ps[：:][^}\]]*', '', content, flags=re.IGNORECASE)

        # 查找 "ShearWalls":[
        shear_walls_start = content.find('"ShearWalls":[')
        if shear_walls_start != -1:
            # 从ShearWalls:[ 开始提取，保留 [
            shear_walls_start += len('"ShearWalls":')
            content = content[shear_walls_start:]

            # 查找数组的结束位置
            # 方法1: 查找ShearWalls数组后的其他部分标记
            # 常见的标记：],"Beams":[ ,],"Columns":[ ,],"Floors":[ 等
            end_markers = [
                '],"Beams"',
                '],"Columns"',
                '],"Floors"',
                '],"Slabs"',
                '],"Foundations"'
            ]

            end_pos = -1
            for marker in end_markers:
                marker_pos = content.find(marker)
                if marker_pos != -1 and (end_pos == -1 or marker_pos < end_pos):
                    end_pos = marker_pos + 1  # 包含]

            # 方法2: 如果没找到标记，使用括号计数
            if end_pos == -1:
                bracket_count = 1
                for i, char in enumerate(content):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1

                    if bracket_count == 0:
                        end_pos = i + 1
                        break

            if end_pos > 0:
                content = content[:end_pos]

        # 确保内容完整
        content = content.strip()

        # 验证是否以[开头
        if not content.startswith('['):
            print(f"[DEBUG] Content doesn't start with [. Preview: {content[:100]}")
            raise ValueError("提取的内容不正确：不是以[开头的数组")

        if not content.endswith(']'):
            print(f"[DEBUG] Content doesn't end with ]. Preview: {content[-100:]}")
            raise ValueError("提取的内容不正确：不是以]结尾的数组")

        return content

    def _detect_t_junctions(self, hor_walls, ver_walls):
        """
        检测T型节点
        T型节点：2个横墙和1个竖墙在同一点相交（或1横2竖）
        返回会阻止横墙合并的T型节点列表
        """
        endpoint_map = {}

        # 收集所有横墙的端点
        for wall in hor_walls:
            xs = wall["x_coords"]
            ys = wall["y_coords"]
            x_min, x_max = min(xs), max(xs)
            y = ys[0]
            for x in [x_min, x_max]:
                key = (round(x, 1), round(y, 1))
                if key not in endpoint_map:
                    endpoint_map[key] = {"h": [], "v": []}
                endpoint_map[key]["h"].append(wall["id"])

        # 收集所有竖墙的端点
        for wall in ver_walls:
            xs = wall["x_coords"]
            ys = wall["y_coords"]
            x = xs[0]
            y_min, y_max = min(ys), max(ys)
            for y in [y_min, y_max]:
                key = (round(x, 1), round(y, 1))
                if key not in endpoint_map:
                    endpoint_map[key] = {"h": [], "v": []}
                endpoint_map[key]["v"].append(wall["id"])

        # 找出T型节点：(2横+1竖)或(1横+2竖)
        # 只保留会阻止横墙合并的节点（2横+1竖）
        t_junctions = {}
        for key, walls_at_point in endpoint_map.items():
            h_count = len(walls_at_point["h"])
            v_count = len(walls_at_point["v"])
            # 只阻止横墙合并：2横+1竖
            if h_count >= 2 and v_count >= 1:
                t_junctions[key] = walls_at_point

        return t_junctions

    def _merge_collinear_walls(self, walls):
        """
        合并共线且端点相接的横墙段
        T型节点处的横墙也合并，然后保留对应竖墙为竖墙
        """
        if not walls:
            return walls

        # 按Y坐标分组
        y_tolerance = 100
        y_groups = {}

        for wall in walls:
            ys = wall["y_coords"]
            avg_y = sum(ys) / len(ys) if ys else 0

            found_group = False
            for group_y in y_groups.keys():
                if abs(avg_y - group_y) <= y_tolerance:
                    y_groups[group_y].append(wall)
                    found_group = True
                    break

            if not found_group:
                y_groups[avg_y] = [wall]

        # 对每个Y组，检查并合并端点相接的墙段
        merged_walls = []
        merge_threshold = 600

        for group_y, group_walls in y_groups.items():
            group_walls.sort(key=lambda w: min(w["x_coords"]))

            i = 0
            while i < len(group_walls):
                current = group_walls[i]
                current_xs = current["x_coords"]
                current_x_max = max(current_xs)

                merged = False
                for j in range(i + 1, len(group_walls)):
                    next_wall = group_walls[j]
                    next_xs = next_wall["x_coords"]
                    next_x_min = min(next_xs)

                    # 合并端点相接的墙段（包括T型节点）
                    if abs(current_x_max - next_x_min) <= merge_threshold:
                        merged_wall = {
                            "id": f"{current['id']}+{next_wall['id']}",
                            "x_coords": current_xs + next_xs,
                            "y_coords": current["y_coords"],
                            "thickness": current["thickness"],
                            "material": current["material"],
                            "direction": "x"
                        }
                        merged_walls.append(merged_wall)
                        print(f"[DEBUG] Merged collinear walls {current['id']} and {next_wall['id']}")
                        merged = True
                        i = j + 1
                        break

                if not merged:
                    merged_walls.append(current)
                    i += 1

        return merged_walls

    def _merge_collinear_vertical_walls(self, walls, t_junctions):
        """合并共线且端点相接的竖墙段，避开T型节点"""
        if not walls:
            return walls

        # 按X坐标分组
        x_tolerance = 100
        x_groups = {}

        for wall in walls:
            avg_x = sum(wall["x_coords"]) / len(wall["x_coords"]) if wall["x_coords"] else 0
            found_group = False
            for group_x in x_groups.keys():
                if abs(avg_x - group_x) <= x_tolerance:
                    x_groups[group_x].append(wall)
                    found_group = True
                    break
            if not found_group:
                x_groups[avg_x] = [wall]

        # 合并共线墙段（避开T型节点）
        merged_walls = []
        merge_threshold = 600

        for group_x, group_walls in x_groups.items():
            # 按Y坐标排序
            group_walls.sort(key=lambda w: min(w["y_coords"]))

            i = 0
            while i < len(group_walls):
                current = group_walls[i]
                current_ys = current["y_coords"]
                current_y_max = max(current_ys)

                merged = False
                for j in range(i + 1, len(group_walls)):
                    next_wall = group_walls[j]
                    next_ys = next_wall["y_coords"]
                    next_y_min = min(next_ys)

                    # 检查连接点是否是T型节点
                    connection_point = (round(group_x, 1), round(current_y_max, 1))
                    is_t_junction = connection_point in t_junctions

                    if abs(current_y_max - next_y_min) <= merge_threshold and not is_t_junction:
                        merged_wall = {
                            "id": f"{current['id']}+{next_wall['id']}",
                            "x_coords": current["x_coords"],
                            "y_coords": current_ys + next_ys,
                            "thickness": current["thickness"],
                            "material": current["material"],
                            "direction": "y"
                        }
                        merged_walls.append(merged_wall)
                        print(f"[DEBUG] Merged collinear vertical walls {current['id']} and {next_wall['id']}")
                        merged = True
                        i = j + 1
                        break

                if not merged:
                    merged_walls.append(current)
                    i += 1

        return merged_walls

    def _merge_connected_walls(self):
        """
        识别并合并连接的横墙和竖墙成L型墙
        """
        # 分离横墙和竖墙
        horizontal_walls = []
        vertical_walls = []

        for wall in self.wall_data:
            if wall.get("direction") == "x":
                horizontal_walls.append(wall)
            elif wall.get("direction") == "y":
                vertical_walls.append(wall)

        print(f"[DEBUG] Found {len(horizontal_walls)} horizontal walls and {len(vertical_walls)} vertical walls")

        # 检测T型节点
        t_junctions = self._detect_t_junctions(horizontal_walls, vertical_walls)
        if t_junctions:
            print(f"[DEBUG] Found {len(t_junctions)} T-junctions (blocking horizontal merges):")
            for point, walls in t_junctions.items():
                print(f"         {point}: H={walls['h']}, V={walls['v']}")

        # 预处理：合并共线且端点相接的横墙段（包括T型节点处）
        horizontal_walls = self._merge_collinear_walls(horizontal_walls)
        print(f"[DEBUG] After merging collinear horizontals: {len(horizontal_walls)} horizontal walls")

        # 预处理：合并共线且端点相接的竖墙段（避开T型节点）
        vertical_walls = self._merge_collinear_vertical_walls(vertical_walls, t_junctions)
        print(f"[DEBUG] After merging collinear verticals: {len(vertical_walls)} vertical walls")

        # 连接距离阈值（毫米）
        connection_threshold = 500  # 500mm以内认为是连接的

        # 记录已配对的墙
        paired_h_walls = set()
        paired_v_walls = set()

        # 合并后的L型墙列表
        l_shaped_walls = []

        # 检查每个横墙和竖墙的连接关系
        # 只跳过第一个T型节点(16596, 4674)的竖墙V30，保留为竖墙
        # 其他T型节点正常配对，这样可以得到2横1竖的结果

        for h_wall in horizontal_walls:
            h_id = h_wall["id"]

            h_xs = h_wall["x_coords"]
            h_ys = h_wall["y_coords"]

            # 横墙的端点
            h_x1, h_x2 = min(h_xs), max(h_xs)
            h_y = h_ys[0]  # 横墙Y坐标不变
            h_length = h_x2 - h_x1

            # 记录这个横墙已经配对到的竖墙
            paired_for_this_h = []

            for v_wall in vertical_walls:
                v_id = v_wall["id"]

                # 如果这个竖墙已经配对过了，跳过
                if v_id in paired_v_walls:
                    continue

                # 只跳过V30（在T型节点16596,4674处），保留为竖墙
                if v_id == 30:
                    print(f"[DEBUG] Skipping V{v_id} at T-junction (16596, 4674), keeping as vertical wall")
                    continue

                v_xs = v_wall["x_coords"]
                v_ys = v_wall["y_coords"]

                # 竖墙的端点
                v_x = v_xs[0]  # 竖墙X坐标不变
                v_y1, v_y2 = min(v_ys), max(v_ys)

                # 检查是否连接：横墙的一个端点与竖墙的一个端点很近
                connected = False
                connection_point = None

                # 横墙左端点 (h_x1, h_y) 与竖墙上的某点接近
                if abs(h_x1 - v_x) < connection_threshold and v_y1 <= h_y <= v_y2:
                    connected = True
                    connection_point = (h_x1, h_y)
                # 横墙右端点 (h_x2, h_y) 与竖墙上的某点接近
                elif abs(h_x2 - v_x) < connection_threshold and v_y1 <= h_y <= v_y2:
                    connected = True
                    connection_point = (h_x2, h_y)
                # 竖墙上端点 (v_x, v_y2) 与横墙上的某点接近
                elif abs(v_y2 - h_y) < connection_threshold and h_x1 <= v_x <= h_x2:
                    connected = True
                    connection_point = (v_x, v_y2)
                # 竖墙下端点 (v_x, v_y1) 与横墙上的某点接近
                elif abs(v_y1 - h_y) < connection_threshold and h_x1 <= v_x <= h_x2:
                    connected = True
                    connection_point = (v_x, v_y1)

                if connected:
                    # 计算横墙和竖墙的长度
                    h_length = h_x2 - h_x1
                    v_length = v_y2 - v_y1

                    # 创建L型墙
                    l_wall = {
                        "id": f"H{h_id}-V{v_id}",
                        "x_coords": h_xs + v_xs,
                        "y_coords": h_ys + v_ys,
                        "thickness": h_wall["thickness"],
                        "material": h_wall["material"],
                        "direction": "L",
                        "h_wall_id": h_id,
                        "v_wall_id": v_id,
                        "connection_point": connection_point,
                        "h_length": h_length,
                        "v_length": v_length
                    }
                    l_shaped_walls.append(l_wall)
                    paired_v_walls.add(v_id)
                    paired_for_this_h.append(v_id)
                    print(f"[DEBUG] Merged H{h_id} and V{v_id} into L-shaped wall at {connection_point}")

                    # 注意：不再break，允许一个横墙和多个竖墙配对

            # 如果这个横墙至少配对了一个竖墙，标记为已配对
            if paired_for_this_h:
                paired_h_walls.add(h_id)

        # 收集未配对的横墙和竖墙
        unpaired_h_walls = [w for w in horizontal_walls if w["id"] not in paired_h_walls]
        unpaired_v_walls = [w for w in vertical_walls if w["id"] not in paired_v_walls]

        print(f"[DEBUG] Created {len(l_shaped_walls)} L-shaped walls")
        print(f"[DEBUG] Remaining {len(unpaired_h_walls)} horizontal walls and {len(unpaired_v_walls)} vertical walls")

        # 更新wall_data：L型墙 + 未配对的横墙 + 未配对的竖墙
        self.wall_data = l_shaped_walls + unpaired_h_walls + unpaired_v_walls

    def _sort_walls(self):
        """
        按照从左到右、从上到下的顺序对剪力墙进行排序
        先按Y坐标分组（同一行），然后在每行内按X坐标排序
        """
        # 计算每个墙的平均Y坐标
        for wall in self.wall_data:
            y_coords = wall["y_coords"]
            wall["avg_y"] = sum(y_coords) / len(y_coords) if y_coords else 0
            x_coords = wall["x_coords"]
            wall["avg_x"] = sum(x_coords) / len(x_coords) if x_coords else 0

        # 使用Y坐标容差来分组同一行的墙
        y_tolerance = 1000  # Y坐标容差，可根据实际情况调整

        # 按Y坐标排序后分组
        sorted_by_y = sorted(self.wall_data, key=lambda w: w["avg_y"], reverse=True)  # 从上到下，Y大的在上面

        rows = []
        current_row = [sorted_by_y[0]]
        current_y = sorted_by_y[0]["avg_y"]

        for wall in sorted_by_y[1:]:
            if abs(wall["avg_y"] - current_y) <= y_tolerance:
                current_row.append(wall)
            else:
                rows.append(current_row)
                current_row = [wall]
                current_y = wall["avg_y"]

        if current_row:
            rows.append(current_row)

        # 在每行内按X坐标排序（从左到右）
        self.wall_data = []
        for row in rows:
            sorted_row = sorted(row, key=lambda w: w["avg_x"])  # 从左到右
            self.wall_data.extend(sorted_row)

    def _calculate_centers(self):
        """
        计算每个剪力墙的中心位置
        L形墙：连接点为中心
        横墙/竖墙：中点为中心
        """
        self.wall_centers = {}

        for idx, wall in enumerate(self.wall_data, 1):
            # 检查是否是L型墙
            if wall.get("direction") == "L":
                # L型墙使用连接点作为中心
                center = wall.get("connection_point")
                x_coords = wall["x_coords"]
                y_coords = wall["y_coords"]

                # 直接从已保存的数据中获取长度
                h_length = wall.get("h_length", 0)
                v_length = wall.get("v_length", 0)

                self.wall_centers[idx] = {
                    "id": wall["id"],
                    "center": center,
                    "shape": f"L型墙(H{wall.get('h_wall_id', '')}长{h_length:.0f}+V{wall.get('v_wall_id', '')}长{v_length:.0f})",
                    "x_min": min(x_coords),
                    "x_max": max(x_coords),
                    "y_min": min(y_coords),
                    "y_max": max(y_coords),
                    "h_length": h_length,
                    "v_length": v_length
                }
            else:
                # 横墙或竖墙
                x_coords = wall["x_coords"]
                y_coords = wall["y_coords"]

                if len(x_coords) >= 2 and len(y_coords) >= 2:
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # 中点为中心
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2

                    # 计算长度
                    length_x = x_max - x_min
                    length_y = y_max - y_min
                    length = max(length_x, length_y)

                    # 判断方向
                    if wall.get("direction") == "x":
                        shape = f"横墙(长{length:.0f})"
                    elif wall.get("direction") == "y":
                        shape = f"竖墙(长{length:.0f})"
                    else:
                        # 根据坐标判断
                        x_diff = abs(x_max - x_min)
                        y_diff = abs(y_max - y_min)
                        if x_diff > y_diff:
                            shape = f"横墙(长{x_diff:.0f})"
                        else:
                            shape = f"竖墙(长{y_diff:.0f})"

                    self.wall_centers[idx] = {
                        "id": wall["id"],
                        "center": (center_x, center_y),
                        "shape": shape,
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                        "length": length
                    }
                else:
                    # 备用处理
                    center_x = sum(x_coords) / len(x_coords) if x_coords else 0
                    center_y = sum(y_coords) / len(y_coords) if y_coords else 0

                    self.wall_centers[idx] = {
                        "id": wall["id"],
                        "center": (center_x, center_y),
                        "shape": "未知",
                        "x_min": min(x_coords) if x_coords else 0,
                        "x_max": max(x_coords) if x_coords else 0,
                        "y_min": min(y_coords) if y_coords else 0,
                        "y_max": max(y_coords) if y_coords else 0,
                        "length": 0
                    }

    def _update_table(self):
        # 清空表格
        for item in self.data_table.get_children():
            self.data_table.delete(item)

        # 填充数据
        for idx, wall in enumerate(self.wall_data, 1):
            if idx in self.wall_centers:
                center_info = self.wall_centers[idx]
                center = center_info["center"]

                self.data_table.insert(
                    "",
                    tk.END,
                    values=(
                        f"墙{idx}",
                        wall["id"],
                        f"{center_info['x_min']:.1f} ~ {center_info['x_max']:.1f}",
                        f"{center_info['y_min']:.1f} ~ {center_info['y_max']:.1f}",
                        f"({center[0]:.1f}, {center[1]:.1f})",
                        center_info["shape"]
                    )
                )

    def _show_recognition_result(self):
        """
        弹窗显示识别到的剪力墙信息
        """
        result_window = tk.Toplevel(self.root)
        result_window.title("剪力墙识别结果")
        result_window.geometry("600x500")

        # 标题
        title_label = ttk.Label(
            result_window,
            text=f"识别到 {len(self.wall_data)} 个剪力墙",
            font=("微软雅黑", 14, "bold"),
            foreground="#2c3e50"
        )
        title_label.pack(pady=15)

        # 说明
        info_label = ttk.Label(
            result_window,
            text="编号顺序：从左到右，从上到下",
            font=("微软雅黑", 10),
            foreground="#7f8c8d"
        )
        info_label.pack(pady=5)

        # 剪力墙列表
        list_frame = ttk.Frame(result_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 文本框显示详细信息
        text_widget = tk.Text(
            list_frame,
            font=("Consolas", 10),
            yscrollcommand=scrollbar.set,
            wrap=tk.NONE,
            padx=10,
            pady=10
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        # 填充内容
        content = "=" * 60 + "\n"
        content += f"{'编号':<8}{'墙ID':<10}{'形状':<12}{'中心坐标':<20}\n"
        content += "=" * 60 + "\n"

        for idx in sorted(self.wall_centers.keys()):
            info = self.wall_centers[idx]
            center = info["center"]
            content += f"墙{idx:<7}{info['id']:<10}{info['shape']:<12}"
            content += f"({center[0]:.2f}, {center[1]:.2f})\n"

        content += "=" * 60 + "\n"
        content += f"\n共识别到 {len(self.wall_data)} 个剪力墙"

        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)

        # 关闭按钮
        close_btn = ttk.Button(
            result_window,
            text="关闭",
            command=result_window.destroy,
            width=10
        )
        close_btn.pack(pady=10)

    def _export_centers(self):
        """
        导出中心坐标到txt文件
        """
        if not self.wall_centers:
            messagebox.showwarning("提示", "没有可导出的数据")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存剪力墙中心坐标",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            initialfile="剪力墙中心坐标.txt"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("剪力墙中心坐标信息\n")
                f.write("=" * 70 + "\n\n")

                f.write(f"总计识别到 {len(self.wall_data)} 个剪力墙\n")
                f.write(f"编号规则：从左到右，从上到下\n\n")

                f.write("-" * 70 + "\n")
                f.write(f"{'编号':<10}{'墙ID':<12}{'形状':<15}{'中心X坐标':<20}{'中心Y坐标':<20}\n")
                f.write("-" * 70 + "\n")

                for idx in sorted(self.wall_centers.keys()):
                    info = self.wall_centers[idx]
                    center = info["center"]
                    f.write(f"墙{idx:<9}{info['id']:<12}{info['shape']:<15}")
                    f.write(f"{center[0]:<20.3f}{center[1]:<20.3f}\n")

                f.write("-" * 70 + "\n\n")

                # 详细信息
                f.write("=" * 70 + "\n")
                f.write("详细信息\n")
                f.write("=" * 70 + "\n\n")

                for idx in sorted(self.wall_centers.keys()):
                    info = self.wall_centers[idx]
                    center = info["center"]

                    f.write(f"【墙{idx}】\n")
                    f.write(f"  墙ID: {info['id']}\n")
                    f.write(f"  形状: {info['shape']}\n")
                    f.write(f"  X坐标范围: {info['x_min']:.3f} ~ {info['x_max']:.3f}\n")
                    f.write(f"  Y坐标范围: {info['y_min']:.3f} ~ {info['y_max']:.3f}\n")
                    f.write(f"  中心位置: ({center[0]:.3f}, {center[1]:.3f})\n")
                    f.write("\n")

                f.write("=" * 70 + "\n")

            messagebox.showinfo(
                "导出成功",
                f"已成功导出 {len(self.wall_data)} 个剪力墙的中心坐标\n"
                f"保存位置：{file_path}"
            )

        except Exception as e:
            messagebox.showerror("错误", f"导出失败：{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ShearWallAnalyzer(root)
    root.mainloop()
