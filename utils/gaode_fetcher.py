"""
高德地图交通数据采集器
=====================
从高德开放平台"交通态势"API 采集国内城市路况数据，
并将其处理为与 PEMS04 相同格式的 numpy 数组，供 RL-TDAGCN 训练使用。

高德交通态势 API 文档: https://lbs.amap.com/api/trafficapi/guides/trafficinfo
API 端点: https://restapi.amap.com/v3/traffic/status/rectangle

数据流:
    高德API(每次采集) -> traffic_raw.json (原始记录)
    -> process_to_numpy() -> traffic.npy (T,N,3) + adj.csv + node_names.json

说明:
    - 高德交通态势API属于"高级服务接口"，免费配额仅约100次/天(非2000次)
    - 5min间隔与天气采集器一致, 保证时间戳对齐
    - 配额超限时脚本自动等待次日0点重置后继续采集
    - 添加文件锁防止多终端并发采集导致配额浪费
    - 所有 API 调用均有重试机制，应对网络抖动
    - 不硬编码 API Key，从 config 或环境变量读取

v5改进 (2026-08-04):
    - 新增多Key分时段轮转采集器 MultiKeyGaodeFetcher
    - 4个API Key各负责6小时时段, 合计288次/天 = 全天5分钟完整覆盖
    - 解决单Key仅能采集6.7小时/天的配额瓶颈
    - 支持7-14天长时间采集, 时间段可自定义
"""
import os
import sys
import json
import time
import math
import logging
import requests
import numpy as np
import pandas as pd

# 文件锁: 防止多终端同时运行采集器，避免并发消耗API配额
# 跨平台兼容: Windows用msvcrt, Linux用fcntl
import platform
if platform.system() == 'Windows':
    import msvcrt as _fcntl_lock
    _IS_WINDOWS = True
else:
    import fcntl as _fcntl_lock
    _IS_WINDOWS = False

# 配置日志格式，方便研究生查看采集过程
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 高德交通态势 API 端点（矩形区域查询）
GAODE_TRAFFIC_URL = 'https://restapi.amap.com/v3/traffic/status/rectangle'

# 高德交通态势API每日免费配额
# 实测: 2026-08-02采到94条即超限, 2026-08-03采到100条即超限
# 高德官方文档标注此接口为"高级服务接口", 免费配额远低于普通Web服务
# 保守取 80, 留出余量防止偶发重试消耗
GAODE_DAILY_QUOTA = 80


class GaodeTrafficFetcher:
    """
    高德交通数据采集器

    负责从高德交通态势 API 采集路况数据，并转换为模型可用的 numpy 格式。

    使用示例:
        fetcher = GaodeTrafficFetcher(
            api_key=os.environ['GAODE_API_KEY'],
            rectangle='116.25,39.80;116.55,40.05',
            interval=300
        )
        # 单次采集
        roads, ts = fetcher.fetch_once()
        # 持续采集 14 天
        fetcher.collect_continuous(duration_hours=24*14, output_dir='data/gaode')
        # 转为 numpy
        fetcher.process_to_numpy('data/gaode/traffic_raw.json', 'data/gaode')
    """

    def __init__(self, api_key, rectangle, interval=300, level=5, max_retry=3, timeout=10):
        """
        Args:
            api_key: 高德开放平台 API Key（Web 服务类型）
            rectangle: 采集矩形区域，格式 "经度1,纬度1;经度2,纬度2"
                       例如北京四环附近: '116.25,39.80;116.55,40.05'
                       注意: 区域不宜过大，单次请求道路数有上限
            interval: 采集间隔(秒)，默认 300s=5min，与 PEMS04 频率一致
            level: 道路级别，5=返回所有级别道路（含高速/主干道/次干道等）
            max_retry: API 调用失败时的最大重试次数
            timeout: 单次请求超时时间(秒)
        """
        if not api_key:
            raise ValueError(
                '高德 API Key 不能为空！请通过环境变量 GAODE_API_KEY 或 config 传入。'
            )
        self.api_key = api_key
        self.rectangle = rectangle
        self.interval = interval
        self.level = level
        self.max_retry = max_retry
        self.timeout = timeout

        # 当日已调用次数（用于频率控制，防止超免费配额）
        self._daily_count = 0
        self._daily_count_date = time.strftime('%Y-%m-%d')

        logger.info(
            f'GaodeTrafficFetcher 初始化: 区域={rectangle}, '
            f'间隔={interval}s, 级别={level}'
        )

    # ------------------------------------------------------------------
    # 频率控制
    # ------------------------------------------------------------------
    def _check_and_increment_quota(self):
        """
        检查当日调用配额，若已超上限则抛出异常。
        每跨天重置计数。
        """
        today = time.strftime('%Y-%m-%d')
        if today != self._daily_count_date:
            # 跨天，重置计数
            self._daily_count = 0
            self._daily_count_date = today
        if self._daily_count >= GAODE_DAILY_QUOTA:
            raise RuntimeError(
                f'今日高德 API 调用已达上限({GAODE_DAILY_QUOTA}次)，请明日再试。'
            )
        self._daily_count += 1

    # ------------------------------------------------------------------
    # 单次采集
    # ------------------------------------------------------------------
    def fetch_once(self):
        """
        单次采集交通数据

        通过高德交通态势 API 查询矩形区域内的路况，返回结构化道路列表。

        Returns:
            roads_data: list[dict]，每条道路包含字段:
                - name: 道路名称
                - status: 路况状态 (1畅通/2缓行/3拥堵/4严重拥堵)
                - speed: 平均速度(km/h)
                - direction: 行驶方向描述
                - coords: 坐标点列表 [[lng, lat], ...]
            timestamp: 采集时间戳(秒级 UNIX 时间)

        Raises:
            RuntimeError: API 返回错误码或当日配额已用尽
        """
        # 1. 配额检查
        self._check_and_increment_quota()

        # 2. 构造请求参数
        #    extensions=all 返回详细道路信息；level=5 返回所有级别道路
        params = {
            'key': self.api_key,
            'rectangle': self.rectangle,
            'extensions': 'all',
            'level': self.level,
            'output': 'json',
        }

        # 3. 带重试的请求（应对网络抖动、限流）
        last_exc = None
        for attempt in range(1, self.max_retry + 1):
            try:
                resp = requests.get(
                    GAODE_TRAFFIC_URL, params=params, timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, ValueError) as e:
                last_exc = e
                logger.warning(
                    f'第 {attempt}/{self.max_retry} 次请求失败: {e}，'
                    f'{attempt*2}s 后重试...'
                )
                time.sleep(attempt * 2)  # 指数退避
        else:
            raise RuntimeError(f'高德 API 请求失败，已达最大重试次数: {last_exc}')

        # 4. 校验返回码
        #    高德 status=1 表示成功，其余为失败
        if data.get('status') != '1':
            err_code = data.get('infocode', 'unknown')
            err_msg = data.get('info', 'unknown')
            raise RuntimeError(
                f'高德 API 返回错误: code={err_code}, msg={err_msg}'
            )

        # 5. 解析道路数据
        timestamp = int(time.time())
        roads_data = self._parse_roads(data)
        logger.info(
            f'采集成功: 共 {len(roads_data)} 条道路, '
            f'今日已用 {self._daily_count}/{GAODE_DAILY_QUOTA}'
        )
        return roads_data, timestamp

    @staticmethod
    def _parse_roads(api_response):
        """
        从高德 API 返回的 JSON 中解析道路信息

        高德返回结构:
            trafficinfo -> roads[] -> 每条路有
            name/status/direction/speed/polyline(坐标串)
            trafficinfo -> evaluation -> 总体路况统计(expedite/congested等, 无roads)

        Args:
            api_response: 高德 API 返回的完整 JSON 字典

        Returns:
            list[dict]: 标准化后的道路数据
        """
        roads_data = []
        try:
            # roads 列表在 trafficinfo 下 (不是 evaluation 下)
            traffic_info = api_response.get('trafficinfo', {})
            roads = traffic_info.get('roads', [])
        except Exception as e:
            logger.error(f'解析高德返回结构失败: {e}')
            return roads_data

        for road in roads:
            # 高德 API 返回的字段名为 polyline (坐标串)，非 polycode
            # 格式: "lng1,lat1;lng2,lat2;..."
            polycode = road.get('polyline', '') or road.get('polycode', '')
            coords = []
            if polycode:
                for pt in polycode.split(';'):
                    pt = pt.strip()
                    if not pt:
                        continue
                    parts = pt.split(',')
                    if len(parts) == 2:
                        try:
                            coords.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            continue

            # 解析速度（可能为空字符串，做容错）
            try:
                speed = float(road.get('speed', 0))
            except (TypeError, ValueError):
                speed = 0.0

            roads_data.append({
                'name': road.get('name', f'unknown_{len(roads_data)}'),
                'status': int(road.get('status', 1) or 1),
                'speed': speed,
                'direction': road.get('direction', ''),
                'coords': coords,
            })
        return roads_data

    # ------------------------------------------------------------------
    # 图结构构建
    # ------------------------------------------------------------------
    def build_graph(self, roads_data):
        """
        从采集到的道路数据构建图结构

        构图策略:
            - 每条道路映射为一个节点
            - 邻接关系: 若两条道路的端点坐标足够近(阈值 connect_threshold 米)，
              则认为它们相连，边权=两点间距离(米)
            - 距离通过 haversine 公式计算（考虑地球曲率）

        这样生成的 adj_list 与 PEMS04.csv 的 (from, to, cost) 格式完全兼容，
        后续可由 load_adj_from_csv() 转为高斯核邻接矩阵。

        Args:
            roads_data: list[dict]，fetch_once() 的返回

        Returns:
            node_names: list[str]，道路名称列表（索引即节点ID）
            adj_list: list[(from_idx, to_idx, cost)]，边列表
            speeds: list[float]，各道路当前速度
            statuses: list[int]，各道路拥堵状态(1-4)
        """
        node_names = [r['name'] for r in roads_data]
        speeds = [r['speed'] for r in roads_data]
        statuses = [r['status'] for r in roads_data]

        # 提取每条道路的端点坐标（取首尾两点用于判断连通性）
        # 若道路只有一组坐标，则用该点作为端点
        endpoints = []
        for r in roads_data:
            coords = r['coords']
            if len(coords) >= 2:
                endpoints.append((coords[0], coords[-1]))  # (起点, 终点)
            elif len(coords) == 1:
                endpoints.append((coords[0], coords[0]))
            else:
                # 无坐标的道路，置为 None，后续跳过
                endpoints.append(None)

        # 连通性判断阈值(米): 端点距离小于此值则认为道路相连
        # 200m 对应实际路网中相邻路口的典型间距
        connect_threshold = 200.0

        adj_list = []
        n = len(roads_data)
        for i in range(n):
            if endpoints[i] is None:
                continue
            for j in range(n):
                if i == j or endpoints[j] is None:
                    continue
                # 取 i 的所有端点与 j 的所有端点的最小距离
                min_dist = float('inf')
                for p1 in endpoints[i]:
                    for p2 in endpoints[j]:
                        d = self._haversine_distance(p1, p2)
                        if d < min_dist:
                            min_dist = d
                # 距离小于阈值则建边（双向，cost=距离）
                if min_dist < connect_threshold:
                    # cost 保留为距离，与 PEMS04.csv 的 cost 语义一致
                    adj_list.append((i, j, round(min_dist, 2)))

        logger.info(
            f'构建图: 节点数={n}, 边数={len(adj_list)}, '
            f'连通阈值={connect_threshold}m'
        )
        return node_names, adj_list, speeds, statuses

    @staticmethod
    def _haversine_distance(p1, p2):
        """
        用 haversine 公式计算两个经纬度坐标之间的球面距离(米)

        Args:
            p1: [lng, lat]
            p2: [lng, lat]

        Returns:
            距离(米)
        """
        R = 6371000.0  # 地球半径(米)
        lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
        lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    # ------------------------------------------------------------------
    # 持续采集
    # ------------------------------------------------------------------
    def collect_continuous(self, duration_hours, output_dir):
        """
        持续采集交通数据

        每隔 interval 秒采集一次，持续 duration_hours 小时。
        所有原始结果追加保存到 traffic_raw.json。

        采集期间若某次失败，记录错误并继续下一次（不中断整体流程）。
        同时维护一个可恢复的增量文件，避免中途崩溃丢失已采数据。

        Args:
            duration_hours: 采集持续小时数
            output_dir: 数据保存目录
        """
        os.makedirs(output_dir, exist_ok=True)
        raw_path = os.path.join(output_dir, 'traffic_raw.json')

        # === [v4新增] 文件锁: 防止多终端并发采集 (v5跨平台兼容) ===
        # 多个终端同时运行会导致API配额成倍消耗且数据文件并发写入损坏
        lock_path = os.path.join(output_dir, '.collection.lock')
        lock_fd = None
        try:
            lock_fd = open(lock_path, 'w')
            try:
                if _IS_WINDOWS:
                    _fcntl_lock.locking(lock_fd.fileno(), _fcntl_lock.LK_NBLCK, 1)
                else:
                    _fcntl_lock.flock(lock_fd.fileno(), _fcntl_lock.LOCK_EX | _fcntl_lock.LOCK_NB)
            except (OSError, IOError):
                logger.error(
                    f'检测到另一个采集进程正在运行 (锁文件: {lock_path})。\n'
                    f'请勿同时开多个终端采集交通数据，这会浪费API配额！\n'
                    f'如确信无其他进程, 删除锁文件后重试。'
                )
                return
            logger.info(f'获取文件锁成功, 开始独占采集')
        except Exception as e:
            logger.warning(f'文件锁初始化失败(非致命, 继续): {e}')

        # 支持断点续采: 若已有数据文件，从中恢复
        if os.path.exists(raw_path):
            with open(raw_path, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
            logger.info(f'检测到已有 {len(all_records)} 条历史记录，将追加采集')
        else:
            all_records = []

        total_seconds = duration_hours * 3600
        total_count = max(1, int(total_seconds // self.interval))
        logger.info(
            f'开始持续采集: 时长={duration_hours}h, 间隔={self.interval}s, '
            f'预计采集 {total_count} 次'
        )

        start_time = time.time()
        success_count = 0
        fail_count = 0
        consecutive_quota_errors = 0  # 连续配额错误计数

        for i in range(total_count):
            try:
                roads_data, ts = self.fetch_once()
                all_records.append({
                    'timestamp': ts,
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)),
                    'roads': roads_data,
                })
                success_count += 1
                consecutive_quota_errors = 0  # 成功则重置计数

                # 每采集 10 次落盘一次，防止崩溃丢失
                if (i + 1) % 10 == 0:
                    self._save_records(raw_path, all_records)
                    logger.info(
                        f'进度: {i+1}/{total_count} '
                        f'(成功 {success_count}, 失败 {fail_count})'
                    )

            except Exception as e:
                fail_count += 1
                error_str = str(e)

                # [v3新增] 检测配额超限错误，智能等待次日重置
                if '10044' in error_str or 'USER_DAILY_QUERY_OVER_LIMIT' in error_str:
                    consecutive_quota_errors += 1
                    logger.error(f'第 {i+1} 次采集失败: API日配额已耗尽 ({error_str})')

                    if consecutive_quota_errors >= 3:
                        # 连续3次配额错误，计算到次日0点的等待时间
                        now = time.localtime()
                        # 次日0点的时间戳
                        import datetime
                        tomorrow = datetime.datetime(now.tm_year, now.tm_mon, now.tm_mday) + datetime.timedelta(days=1)
                        wait_until = tomorrow.timestamp()
                        wait_seconds = wait_until - time.time()

                        if wait_seconds > 0 and wait_seconds < 86400:
                            logger.info(
                                f'连续{consecutive_quota_errors}次配额超限， '
                                f'暂停采集，等待次日0点配额重置 '
                                f'(预计等待 {wait_seconds/3600:.1f} 小时)'
                            )
                            # 先保存当前数据
                            self._save_records(raw_path, all_records)
                            # 等待到次日0点
                            time.sleep(wait_seconds + 60)  # 多等1分钟确保重置
                            consecutive_quota_errors = 0
                            logger.info('配额已重置，恢复采集')
                        else:
                            logger.error('无法计算等待时间，终止采集')
                            break
                else:
                    logger.error(f'第 {i+1} 次采集失败: {e}')
                    # 普通失败后短暂等待
                    time.sleep(5)

            # 等待到下一次采集时刻
            # 若某次采集耗时较长，则减少等待时间以保持节奏
            elapsed = time.time() - start_time
            next_target = (i + 1) * self.interval
            wait = next_target - elapsed
            if wait > 0:
                time.sleep(wait)

        # 最终落盘
        self._save_records(raw_path, all_records)
        logger.info(
            f'采集完成! 总计 {len(all_records)} 条记录 '
            f'(本次成功 {success_count}, 失败 {fail_count}), 保存至 {raw_path}'
        )

        # === [v4新增] 释放文件锁 (v5跨平台兼容) ===
        if lock_fd is not None:
            try:
                if _IS_WINDOWS:
                    _fcntl_lock.locking(lock_fd.fileno(), _fcntl_lock.LK_UNLCK, 1)
                else:
                    _fcntl_lock.flock(lock_fd.fileno(), _fcntl_lock.LOCK_UN)
                lock_fd.close()
                if os.path.exists(lock_path):
                    os.remove(lock_path)
                logger.info('文件锁已释放')
            except Exception:
                pass

    @staticmethod
    def _save_records(path, records):
        """将记录列表写入 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 转换为 numpy 格式
    # ------------------------------------------------------------------
    def process_to_numpy(self, raw_data_path, output_dir):
        """
        将原始 JSON 数据处理为 numpy 格式，与 PEMS04 完全兼容

        处理逻辑:
            1. 从第一次采集建立稳定的节点映射(道路名称->索引)
               （保证不同时间步的节点顺序一致）
            2. 每个时间步提取 [flow, occupancy, speed] 三特征:
               - flow:      高德无流量，用 speed*2 近似（量级合理）
               - occupancy: 用拥堵状态归一化值替代 (status 1->0.2, 2->0.5, 3->0.8, 4->1.0)
               - speed:     高德平均速度(km/h)
            3. 构建邻接边 adj.csv，使用第一次采集的图结构
            4. 缺失道路的速度用上次有效值前向填充

        输出文件:
            - traffic.npy: (T, N, 3) 交通特征数组
            - adj.csv: (from, to, cost) 边列表
            - node_names.json: 道路名称到节点索引的映射
            - meta.json: 采集元信息(节点数/时间步/时间范围)

        Args:
            raw_data_path: traffic_raw.json 路径
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. 加载原始数据
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        if not records:
            raise ValueError('原始数据为空，无法处理')

        logger.info(f'加载原始数据: {len(records)} 个时间步')

        # 2. 建立稳定的节点映射（用第一次有效采集的道路列表）
        first_roads = None
        for rec in records:
            if rec.get('roads'):
                first_roads = rec['roads']
                break
        if first_roads is None:
            raise ValueError('所有时间步的道路数据均为空')

        # 节点映射: 道路名称 -> 索引
        node_names = [r['name'] for r in first_roads]
        name_to_idx = {name: idx for idx, name in enumerate(node_names)}
        N = len(node_names)
        logger.info(f'节点数(道路数): {N}')

        # 3. 构建邻接矩阵（基于第一次采集的图结构，保持稳定）
        _, adj_list, _, _ = self.build_graph(first_roads)

        # 4. 将每个时间步转为 (N, 3) 数组
        T = len(records)
        traffic = np.zeros((T, N, 3), dtype=np.float32)

        # 拥堵状态到占用率的映射（模拟 PEMS04 的 occupancy 概念）
        # status: 1畅通/2缓行/3拥堵/4严重拥堵
        status_to_occ = {1: 0.2, 2: 0.5, 3: 0.8, 4: 1.0}

        # 用于前向填充缺失道路的速度
        last_speed = np.zeros(N, dtype=np.float32)

        for t, rec in enumerate(records):
            roads = rec.get('roads', [])
            # 建立当前时间步 名称->道路 的快速查找
            cur_map = {r['name']: r for r in roads}

            for name, idx in name_to_idx.items():
                if name in cur_map:
                    r = cur_map[name]
                    speed = float(r.get('speed', 0) or 0)
                    status = int(r.get('status', 1) or 1)
                    last_speed[idx] = speed
                else:
                    # 该时间步缺失该道路，用上次有效值
                    speed = last_speed[idx]
                    status = 1  # 默认畅通

                # 三特征: flow=速度*2近似, occupancy=状态映射, speed=速度
                traffic[t, idx, 0] = speed * 2.0          # flow 近似
                traffic[t, idx, 1] = status_to_occ.get(status, 0.2)  # occupancy
                traffic[t, idx, 2] = speed                # speed

        # 5. 保存 traffic.npy
        traffic_path = os.path.join(output_dir, 'traffic.npy')
        np.save(traffic_path, traffic)
        logger.info(f'保存交通数据: {traffic_path}, shape={traffic.shape}')

        # 6. 保存 adj.csv（与 PEMS04.csv 完全相同的列名）
        adj_path = os.path.join(output_dir, 'adj.csv')
        adj_df = pd.DataFrame(adj_list, columns=['from', 'to', 'cost'])
        adj_df.to_csv(adj_path, index=False)
        logger.info(f'保存邻接边: {adj_path}, 边数={len(adj_list)}')

        # 7. 保存节点名称映射
        node_names_path = os.path.join(output_dir, 'node_names.json')
        with open(node_names_path, 'w', encoding='utf-8') as f:
            json.dump(name_to_idx, f, ensure_ascii=False, indent=2)
        logger.info(f'保存节点映射: {node_names_path}')

        # 8. 保存元信息（供 loader 自动检测节点数）
        meta = {
            'num_nodes': N,
            'num_timesteps': T,
            'num_features': 3,
            'feature_names': ['flow_approx', 'occupancy', 'speed'],
            'time_range': {
                'start': records[0].get('datetime'),
                'end': records[-1].get('datetime'),
            },
            'interval_seconds': self.interval,
        }
        meta_path = os.path.join(output_dir, 'meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f'保存元信息: {meta_path}')

        return traffic, adj_list, node_names


class MultiKeyGaodeFetcher:
    """
    [v5新增] 多Key分时段轮转采集器

    解决方案: 高德交通态势API单Key免费配额仅~80次/天, 无法覆盖全天288次(5min间隔)。
    使用4个不同账户的API Key, 每个Key负责6小时时段:
        Key1: 00:00-06:00  (72次, < 80配额)
        Key2: 06:00-12:00  (72次, < 80配额)
        Key3: 12:00-18:00  (72次, < 80配额)
        Key4: 18:00-24:00  (72次, < 80配额)
    合计: 288次/天 = 全天5分钟完整覆盖

    注册方法:
        1. 用4个不同手机号在高德开放平台注册4个账户
        2. 每个账户创建1个"Web服务"类型的Key
        3. 将4个Key填入环境变量或配置文件

    使用示例:
        # 方式1: 环境变量 (推荐)
        #   set GAODE_API_KEY_1=key1
        #   set GAODE_API_KEY_2=key2
        #   set GAODE_API_KEY_3=key3
        #   set GAODE_API_KEY_4=key4
        fetcher = MultiKeyGaodeFetcher.from_env()

        # 方式2: 直接传入Key列表
        fetcher = MultiKeyGaodeFetcher(
            api_keys=['key1', 'key2', 'key3', 'key4'],
            rectangle='116.28,39.95;116.36,40.01',
            interval=300,
        )
        fetcher.collect_continuous(duration_hours=24*14, output_dir='data/gaode')
    """

    # 每个Key负责的时段 (小时, 24小时制)
    # 4个Key各6小时, 每个Key最多用72次 (6h * 12次/h = 72 < 80配额)
    KEY_TIME_SLOTS = [
        (0, 6),   # Key1: 凌晨 00:00-06:00
        (6, 12),  # Key2: 上午 06:00-12:00
        (12, 18), # Key3: 下午 12:00-18:00
        (18, 24), # Key4: 晚间 18:00-24:00
    ]

    def __init__(self, api_keys, rectangle, interval=300, level=5):
        """
        Args:
            api_keys: list[str], 4个高德API Key (按时段顺序: 凌晨/上午/下午/晚间)
            rectangle: 采集矩形区域 "经度1,纬度1;经度2,纬度2"
            interval: 采集间隔(秒), 默认300s=5min
            level: 道路级别, 5=所有级别
        """
        if len(api_keys) < 4:
            raise ValueError(
                f'多Key模式需要至少4个API Key (当前{len(api_keys)}个)。\n'
                f'请注册4个高德账户, 每个账户创建1个Key:\n'
                f'  Key1: 负责 00:00-06:00\n'
                f'  Key2: 负责 06:00-12:00\n'
                f'  Key3: 负责 12:00-18:00\n'
                f'  Key4: 负责 18:00-24:00\n'
                f'注册地址: https://lbs.amap.com/'
            )

        self.api_keys = api_keys[:4]  # 只取前4个
        self.rectangle = rectangle
        self.interval = interval
        self.level = level

        # 为每个Key创建独立的单Key采集器实例
        self.fetchers = []
        for i, key in enumerate(self.api_keys):
            fetcher = GaodeTrafficFetcher(
                api_key=key,
                rectangle=rectangle,
                interval=interval,
                level=level,
            )
            self.fetchers.append(fetcher)
            slot = self.KEY_TIME_SLOTS[i]
            logger.info(f'Key{i+1} 负责时段: {slot[0]:02d}:00-{slot[1]:02d}:00')

        logger.info(
            f'MultiKeyGaodeFetcher 初始化: {len(self.api_keys)}个Key, '
            f'全天{24*60//(interval//60)}次采集, 每Key最多{6*60//(interval//60)}次'
        )

    @classmethod
    def from_env(cls, rectangle='116.28,39.95;116.36,40.01', interval=300):
        """
        从环境变量读取4个API Key创建采集器

        环境变量:
            GAODE_API_KEY_1: 凌晨时段Key (00:00-06:00)
            GAODE_API_KEY_2: 上午时段Key (06:00-12:00)
            GAODE_API_KEY_3: 下午时段Key (12:00-18:00)
            GAODE_API_KEY_4: 晚间时段Key (18:00-24:00)

        兼容: 如果只设置了 GAODE_API_KEY (单Key), 退化为单Key模式
        """
        keys = []
        for i in range(1, 5):
            key = os.environ.get(f'GAODE_API_KEY_{i}', '')
            if key:
                keys.append(key)

        if len(keys) >= 4:
            logger.info(f'从环境变量读取到 {len(keys)} 个Key, 启用多Key轮转模式')
            return cls(keys, rectangle, interval)
        elif len(keys) > 0:
            # 有部分Key但不足4个, 用已有的Key轮换
            logger.warning(
                f'仅读取到 {len(keys)} 个Key (需要4个), '
                f'将用已有Key轮换, 但无法保证全天覆盖'
            )
            # 用已有Key填充到4个
            while len(keys) < 4:
                keys.append(keys[len(keys) % len(keys)])
            return cls(keys, rectangle, interval)
        else:
            # 退化为单Key模式
            single_key = os.environ.get('GAODE_API_KEY', '')
            if not single_key:
                raise EnvironmentError(
                    '未找到高德API Key!\n'
                    '多Key模式: 设置 GAODE_API_KEY_1 ~ GAODE_API_KEY_4\n'
                    '单Key模式: 设置 GAODE_API_KEY\n'
                    '注册地址: https://lbs.amap.com/'
                )
            logger.warning('仅找到单Key, 退化为单Key模式 (每天仅能采集约6.7小时)')
            # 用同一个Key填充4个位置 (实际是单Key模式)
            return cls([single_key] * 4, rectangle, interval)

    def _get_current_key_index(self, current_hour=None):
        """
        根据当前时间确定使用哪个Key

        Args:
            current_hour: 当前小时 (0-23), None则自动获取

        Returns:
            key_index: 0-3, 对应4个Key
        """
        if current_hour is None:
            current_hour = time.localtime().tm_hour

        for i, (start, end) in enumerate(self.KEY_TIME_SLOTS):
            if start <= current_hour < end:
                return i

        return 0  # 默认用Key1 (理论上不会执行到这里)

    def fetch_once(self):
        """
        单次采集: 自动选择当前时段对应的Key

        Returns:
            roads_data, timestamp: 与 GaodeTrafficFetcher.fetch_once 相同
        """
        key_idx = self._get_current_key_index()
        fetcher = self.fetchers[key_idx]
        slot = self.KEY_TIME_SLOTS[key_idx]

        try:
            roads_data, ts = fetcher.fetch_once()
            logger.info(
                f'[多Key] 使用Key{key_idx+1} ({slot[0]:02d}:00-{slot[1]:02d}:00), '
                f'今日该Key已用 {fetcher._daily_count}/{GAODE_DAILY_QUOTA}'
            )
            return roads_data, ts
        except RuntimeError as e:
            error_str = str(e)
            # 如果当前Key配额超限, 尝试用其他Key
            if '上限' in error_str or '10044' in error_str:
                logger.warning(
                    f'Key{key_idx+1} 配额超限, 尝试使用其他Key...'
                )
                for i, alt_fetcher in enumerate(self.fetchers):
                    if i == key_idx:
                        continue
                    try:
                        roads_data, ts = alt_fetcher.fetch_once()
                        logger.info(
                            f'[多Key] Key{key_idx+1}超限, 切换到Key{i+1}成功'
                        )
                        return roads_data, ts
                    except RuntimeError:
                        continue
                # 所有Key都超限
                raise RuntimeError('所有Key的当日配额均已耗尽!')
            raise

    def collect_continuous(self, duration_hours, output_dir):
        """
        持续采集交通数据 (多Key轮转版)

        与单Key版的主要区别:
            - 根据当前时间自动切换Key, 不需要等到次日0点
            - 每天可采集完整288次 (5min间隔), 覆盖全天24小时
            - 单Key仅在自己负责的6小时时段内被调用 (最多72次 < 80配额)

        Args:
            duration_hours: 采集持续小时数 (如 24*14=336 表示14天)
            output_dir: 数据保存目录
        """
        os.makedirs(output_dir, exist_ok=True)
        raw_path = os.path.join(output_dir, 'traffic_raw.json')

        # === 文件锁: 防止多终端并发采集 (v5跨平台兼容) ===
        lock_path = os.path.join(output_dir, '.collection.lock')
        lock_fd = None
        try:
            lock_fd = open(lock_path, 'w')
            try:
                if _IS_WINDOWS:
                    _fcntl_lock.locking(lock_fd.fileno(), _fcntl_lock.LK_NBLCK, 1)
                else:
                    _fcntl_lock.flock(lock_fd.fileno(), _fcntl_lock.LOCK_EX | _fcntl_lock.LOCK_NB)
            except (OSError, IOError):
                logger.error(
                    f'检测到另一个采集进程正在运行 (锁文件: {lock_path})。\n'
                    f'请勿同时开多个终端采集交通数据，这会浪费API配额！\n'
                    f'如确信无其他进程, 删除锁文件后重试。'
                )
                return
            logger.info(f'获取文件锁成功, 开始多Key轮转采集')
        except Exception as e:
            logger.warning(f'文件锁初始化失败(非致命, 继续): {e}')

        # 支持断点续采
        if os.path.exists(raw_path):
            with open(raw_path, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
            logger.info(f'检测到已有 {len(all_records)} 条历史记录，将追加采集')
        else:
            all_records = []

        total_seconds = duration_hours * 3600
        total_count = max(1, int(total_seconds // self.interval))
        logger.info(
            f'开始多Key持续采集: 时长={duration_hours}h ({duration_hours/24:.0f}天), '
            f'间隔={self.interval}s, 预计采集 {total_count} 次\n'
            f'  Key1: 00:00-06:00 (凌晨)\n'
            f'  Key2: 06:00-12:00 (上午)\n'
            f'  Key3: 12:00-18:00 (下午)\n'
            f'  Key4: 18:00-24:00 (晚间)\n'
            f'  每Key每日最多72次, 配额80次/天, 余量充足'
        )

        start_time = time.time()
        success_count = 0
        fail_count = 0

        for i in range(total_count):
            try:
                roads_data, ts = self.fetch_once()
                all_records.append({
                    'timestamp': ts,
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)),
                    'roads': roads_data,
                })
                success_count += 1

                # 每采集10次落盘一次
                if (i + 1) % 10 == 0:
                    self._save_records(raw_path, all_records)
                    # 打印各Key使用情况
                    key_usage = [f._daily_count for f in self.fetchers]
                    logger.info(
                        f'进度: {i+1}/{total_count} '
                        f'(成功{success_count}, 失败{fail_count}) '
                        f'Key用量: {key_usage}'
                    )

            except Exception as e:
                fail_count += 1
                error_str = str(e)

                if '所有Key' in error_str or '10044' in error_str:
                    # 所有Key都超限 (理论上不应发生, 除非Key被额外消耗)
                    logger.error(f'第{i+1}次采集: 所有Key配额耗尽 ({error_str})')
                    # 等待到下一个时段切换 (最多等6小时)
                    now = time.localtime()
                    current_hour = now.tm_hour
                    next_slot_start = ((current_hour // 6) + 1) * 6
                    if next_slot_start >= 24:
                        next_slot_start = 0
                        wait_hours = 24 - current_hour
                    else:
                        wait_hours = next_slot_start - current_hour
                    wait_seconds = wait_hours * 3600 + 60  # 多等1分钟
                    logger.info(f'等待{wait_hours}小时到下一时段...')
                    self._save_records(raw_path, all_records)
                    time.sleep(wait_seconds)
                else:
                    logger.error(f'第{i+1}次采集失败: {e}')
                    time.sleep(5)

            # 等待到下一次采集时刻
            elapsed = time.time() - start_time
            next_target = (i + 1) * self.interval
            wait = next_target - elapsed
            if wait > 0:
                time.sleep(wait)

        # 最终落盘
        self._save_records(raw_path, all_records)
        logger.info(
            f'多Key采集完成! 总计 {len(all_records)} 条记录 '
            f'(本次成功 {success_count}, 失败 {fail_count}), 保存至 {raw_path}'
        )

        # 释放文件锁 (v5跨平台兼容)
        if lock_fd is not None:
            try:
                if _IS_WINDOWS:
                    _fcntl_lock.locking(lock_fd.fileno(), _fcntl_lock.LK_UNLCK, 1)
                else:
                    _fcntl_lock.flock(lock_fd.fileno(), _fcntl_lock.LOCK_UN)
                lock_fd.close()
                if os.path.exists(lock_path):
                    os.remove(lock_path)
                logger.info('文件锁已释放')
            except Exception:
                pass

    @staticmethod
    def _save_records(path, records):
        """将记录列表写入 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def process_to_numpy(self, raw_data_path, output_dir):
        """
        将原始JSON数据转为numpy格式 (委托给单Key采集器处理)
        """
        # 使用第一个fetcher的处理方法 (所有fetcher共用同一套处理逻辑)
        return self.fetchers[0].process_to_numpy(raw_data_path, output_dir)


# ----------------------------------------------------------------------
# 命令行入口：支持单独运行采集
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='高德交通数据采集器 (v5: 多Key轮转)')
    parser.add_argument('--mode', type=str, default='multi',
                        choices=['multi', 'single'],
                        help='采集模式: multi=多Key轮转(推荐), single=单Key(每天仅6.7h)')
    parser.add_argument('--days', type=int, default=14,
                        help='采集天数 (默认14天)')
    parser.add_argument('--rectangle', type=str,
                        default='116.28,39.95;116.36,40.01',
                        help='采集矩形区域 (对角线<10km)')
    parser.add_argument('--interval', type=int, default=300,
                        help='采集间隔秒数 (默认300=5min)')
    parser.add_argument('--output', type=str, default='data/gaode',
                        help='输出目录')
    args = parser.parse_args()

    if args.mode == 'multi':
        # === 多Key模式: 全天5分钟完整覆盖 ===
        print("\n=== 多Key分时段轮转采集模式 ===")
        print("需要4个API Key (4个高德账户):")
        print("  GAODE_API_KEY_1: 00:00-06:00 (凌晨)")
        print("  GAODE_API_KEY_2: 06:00-12:00 (上午)")
        print("  GAODE_API_KEY_3: 12:00-18:00 (下午)")
        print("  GAODE_API_KEY_4: 18:00-24:00 (晚间)")
        print(f"  采集天数: {args.days}天")
        print(f"  间隔: {args.interval}s ({args.interval//60}min)")
        print(f"  每日总采集: {24*3600//args.interval}次 (4Key各{6*3600//args.interval}次)")
        print()

        fetcher = MultiKeyGaodeFetcher.from_env(
            rectangle=args.rectangle,
            interval=args.interval,
        )
    else:
        # === 单Key模式: 每天仅6.7小时 (兼容旧版) ===
        print("\n=== 单Key采集模式 (每天仅约6.7小时数据) ===")
        api_key = os.environ.get('GAODE_API_KEY', '')
        if not api_key:
            raise EnvironmentError(
                '请设置环境变量 GAODE_API_KEY, 或使用 --mode multi 多Key模式'
            )
        fetcher = GaodeTrafficFetcher(
            api_key=api_key,
            rectangle=args.rectangle,
            interval=args.interval,
        )

    # === 首次测试: 验证API连通性 ===
    print("=== 首次采集测试 ===")
    try:
        roads_data, ts = fetcher.fetch_once()
        print(f"  API 连通成功! 返回 {len(roads_data)} 条道路")
        if roads_data:
            for i, r in enumerate(roads_data[:5]):
                print(f"    [{i}] {r['name']} | 状态={r['status']} | 速度={r['speed']}km/h")
        else:
            print("  (当前时段无道路数据，凌晨时段属正常现象)")
        print("  === 测试通过，开始持续采集 ===\n")
    except Exception as e:
        print(f"  API 连接失败: {e}")
        print("  请检查 API Key 和矩形区域参数后再试。")
        sys.exit(1)

    # 持续采集
    fetcher.collect_continuous(
        duration_hours=24 * args.days,
        output_dir=args.output,
    )

    # 转换为numpy格式
    raw_path = os.path.join(args.output, 'traffic_raw.json')
    fetcher.process_to_numpy(raw_path, args.output)
    print(f"\n=== 采集完成! 数据保存至 {args.output} ===")
