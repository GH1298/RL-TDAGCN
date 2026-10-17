"""
和风天气数据采集器
==================
从和风天气(QWeather) API 采集实时天气数据，用于替换 RL-TDAGCN 中全零的天气占位符。

和风天气 API 文档: https://dev.qweather.com/docs/api/weather/weather-now/
API 端点: https://devapi.qweather.com/v7/weather/now

返回 6 维天气向量: [温度, 湿度, 风速, 气压, 降水量, 能见度]
与 config.weather_dim=6 完全对应。

说明:
    - 和风天气免费版每日 1000 次调用额度
    - 若 interval=300s(5min)，一天需 288 次，在免费额度内
    - 本脚本与 gaode_fetcher.py 配合使用，实现交通+天气多模态采集
"""
import os
import json
import time
import logging
import requests
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 和风天气实时天气接口
# 注意: 2026年起 devapi.qweather.com 已逐步停用，需使用控制台中的专属 API Host
# 格式: https://abc1234xyz.def.qweatherapi.com/v7/weather/now
# 请在环境变量 QWEATHER_API_HOST 中设置你的专属 Host (不含 https:// 前缀)
_DEFAULT_QWEATHER_HOST = 'devapi.qweather.com'  # 旧版兜底，新用户必须设置 QWEATHER_API_HOST

# 和风天气免费版每日调用上限（保守值，实际免费版为1000次）
QWEATHER_DAILY_QUOTA = 1000


class WeatherFetcher:
    """
    和风天气数据采集器

    采集实时天气数据，并与交通数据时间对齐，生成 weather_aligned.npy。

    使用示例:
        fetcher = WeatherFetcher(
            api_key=os.environ['QWEATHER_API_KEY'],
            location='101010100'  # 北京
        )
        # 单次获取
        vec, ts = fetcher.fetch_once()
        # 持续采集 24 小时
        fetcher.collect_continuous(duration_hours=24, output_dir='data/gaode')
        # 与交通数据对齐
        fetcher.align_with_traffic(weather_data, traffic_timestamps)
    """

    def __init__(self, api_key, location='101010100', max_retry=3, timeout=10, api_host=None):
        """
        Args:
            api_key: 和风天气 API Key
            location: 城市ID，默认 '101010100'(北京)
                      常用城市ID: 北京101010100, 上海101020100, 广州101280101
                      完整列表见: https://github.com/qwd/LocationList
            max_retry: API 调用失败时的最大重试次数
            timeout: 单次请求超时时间(秒)
            api_host: 和风天气专属 API Host (如 abc1234xyz.def.qweatherapi.com)
                      若为 None，则从环境变量 QWEATHER_API_HOST 读取
                      若环境变量也未设置，则使用旧版 devapi.qweather.com (可能已停用)
        """
        if not api_key:
            raise ValueError(
                '和风天气 API Key 不能为空！请通过环境变量 QWEATHER_API_KEY 或 config 传入。'
            )
        self.api_key = api_key
        self.location = location
        self.max_retry = max_retry
        self.timeout = timeout

        # 确定 API Host: 优先使用参数传入 > 环境变量 > 旧版兜底
        if api_host is None:
            api_host = os.environ.get('QWEATHER_API_HOST', '')
        if not api_host:
            api_host = _DEFAULT_QWEATHER_HOST
            logger.warning(
                '未设置 QWEATHER_API_HOST 环境变量，使用旧版 devapi.qweather.com。\n'
                '  该地址从2026年起逐步停用，如遇 403 错误请前往:\n'
                '  https://console.qweather.com/setting 获取你的专属 API Host\n'
                '  并设置环境变量: $env:QWEATHER_API_HOST = "你的Host"'
            )
        # 构造完整 API URL
        self.api_host = api_host
        self.api_url = f'https://{api_host}/v7/weather/now'

        # 当日已调用次数（用于频率控制）
        self._daily_count = 0
        self._daily_count_date = time.strftime('%Y-%m-%d')

        logger.info(f'WeatherFetcher 初始化: 城市ID={location}, API Host={api_host}')

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
            self._daily_count = 0
            self._daily_count_date = today
        if self._daily_count >= QWEATHER_DAILY_QUOTA:
            raise RuntimeError(
                f'今日和风天气 API 调用已达上限({QWEATHER_DAILY_QUOTA}次)，请明日再试。'
            )
        self._daily_count += 1

    # ------------------------------------------------------------------
    # 单次获取
    # ------------------------------------------------------------------
    def fetch_once(self):
        """
        单次获取天气数据

        调用和风天气 "实时天气(now)" 接口，返回 6 维天气向量。

        Returns:
            weather_vec: list[float] 长度6
                [temp, humidity, wind_speed, pressure, precip, vis]
                - temp:       温度(摄氏度)
                - humidity:   湿度(%)
                - wind_speed: 风速(km/h)
                - pressure:   气压(hPa)
                - precip:     降水量(mm)
                - vis:        能见度(km)
            timestamp: 采集时间戳(秒级 UNIX 时间)

        Raises:
            RuntimeError: API 返回非200状态码或当日配额已用尽
        """
        # 1. 配额检查
        self._check_and_increment_quota()

        # 2. 构造请求参数
        params = {
            'key': self.api_key,
            'location': self.location,
        }

        # 3. 带重试的请求
        last_exc = None
        for attempt in range(1, self.max_retry + 1):
            try:
                resp = requests.get(
                    self.api_url, params=params, timeout=self.timeout
                )
                # 403/401 是认证错误，重试无意义，直接报错
                if resp.status_code in (401, 403):
                    try:
                        err_data = resp.json()
                        err_detail = err_data.get('error', {}).get('detail', '')
                    except (ValueError, AttributeError):
                        err_detail = resp.text[:200]
                    raise RuntimeError(
                        f'和风天气 API 认证失败 (HTTP {resp.status_code}): {err_detail}\n'
                        f'  若为 403 Invalid Host: 请前往 https://console.qweather.com/setting\n'
                        f'  获取你的专属 API Host 并设置环境变量 QWEATHER_API_HOST'
                    )
                resp.raise_for_status()
                data = resp.json()
                break
            except RuntimeError:
                raise  # 认证错误直接抛出，不重试
            except (requests.RequestException, ValueError) as e:
                last_exc = e
                logger.warning(
                    f'第 {attempt}/{self.max_retry} 次请求失败: {e}，'
                    f'{attempt*2}s 后重试...'
                )
                time.sleep(attempt * 2)
        else:
            raise RuntimeError(f'和风天气 API 请求失败，已达最大重试次数: {last_exc}')

        # 4. 校验返回码
        #    和风天气 code='200' 表示成功
        if data.get('code') != '200':
            raise RuntimeError(
                f'和风天气 API 返回错误: code={data.get("code")}, '
                f'msg={data.get("refer", {}).get("reason", "unknown")}'
            )

        # 5. 解析 now 字段
        now = data.get('now', {})
        weather_vec = [
            float(now.get('temp', 0)),            # 温度(摄氏度)
            float(now.get('humidity', 0)),        # 湿度(%)
            float(now.get('windSpeed', 0)),       # 风速(km/h)
            float(now.get('pressure', 0)),        # 气压(hPa)
            float(now.get('precip', 0)),          # 降水量(mm)
            float(now.get('vis', 0)),             # 能见度(km)
        ]
        timestamp = int(time.time())

        logger.info(
            f'采集成功: temp={weather_vec[0]}C, humidity={weather_vec[1]}%, '
            f'wind={weather_vec[2]}km/h, 今日已用 '
            f'{self._daily_count}/{QWEATHER_DAILY_QUOTA}'
        )
        return weather_vec, timestamp

    # ------------------------------------------------------------------
    # 持续采集
    # ------------------------------------------------------------------
    def collect_continuous(self, duration_hours, output_dir, interval=300):
        """
        持续采集天气数据

        每隔 interval 秒采集一次，持续 duration_hours 小时。
        所有原始结果追加保存到 weather_raw.json。

        注意: 和风天气免费版每日 1000 次调用额度。
        若 interval=300s(5min)，一天需 288 次，在免费额度内。

        Args:
            duration_hours: 采集持续小时数
            output_dir: 数据保存目录
            interval: 采集间隔(秒)，默认 300s=5min
        """
        os.makedirs(output_dir, exist_ok=True)
        raw_path = os.path.join(output_dir, 'weather_raw.json')

        # 支持断点续采
        if os.path.exists(raw_path):
            with open(raw_path, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
            logger.info(f'检测到已有 {len(all_records)} 条历史记录，将追加采集')
        else:
            all_records = []

        total_seconds = duration_hours * 3600
        total_count = max(1, int(total_seconds // interval))
        logger.info(
            f'开始持续采集天气: 时长={duration_hours}h, 间隔={interval}s, '
            f'预计采集 {total_count} 次'
        )

        start_time = time.time()
        success_count = 0
        fail_count = 0

        for i in range(total_count):
            try:
                weather_vec, ts = self.fetch_once()
                all_records.append({
                    'timestamp': ts,
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)),
                    'weather': weather_vec,
                })
                success_count += 1

                # 每采集 10 次落盘一次
                if (i + 1) % 10 == 0:
                    self._save_records(raw_path, all_records)
                    logger.info(
                        f'进度: {i+1}/{total_count} '
                        f'(成功 {success_count}, 失败 {fail_count})'
                    )

            except Exception as e:
                fail_count += 1
                logger.error(f'第 {i+1} 次采集失败: {e}')
                time.sleep(5)

            # 等待到下一次采集时刻
            elapsed = time.time() - start_time
            next_target = (i + 1) * interval
            wait = next_target - elapsed
            if wait > 0:
                time.sleep(wait)

        # 最终落盘
        self._save_records(raw_path, all_records)
        logger.info(
            f'采集完成! 总计 {len(all_records)} 条记录 '
            f'(本次成功 {success_count}, 失败 {fail_count}), 保存至 {raw_path}'
        )

    @staticmethod
    def _save_records(path, records):
        """将记录列表写入 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 时间对齐
    # ------------------------------------------------------------------
    @staticmethod
    def align_with_traffic(weather_data, traffic_timestamps):
        """
        将天气数据与交通数据时间对齐

        天气数据采集频率可能与交通数据不同(如天气 10min 一次，交通 5min 一次)，
        需要按交通数据的时间戳找到最近的天气记录，生成 weather_aligned.npy。

        对齐策略: 对每个交通时间戳，选择时间差最小的天气记录，
        生成 (T, 6) 的数组，T = 交通数据时间步数。

        Args:
            weather_data: list[(timestamp, weather_vec)] 天气记录列表
            traffic_timestamps: list[int] 交通数据的时间戳列表

        Returns:
            weather_aligned: (T, 6) numpy 数组，与交通数据时间对齐
        """
        if not weather_data:
            raise ValueError('天气数据为空，无法对齐')
        if not traffic_timestamps:
            raise ValueError('交通时间戳列表为空，无法对齐')

        # 提取天气时间戳和特征，转成 numpy 数组方便向量化计算
        weather_ts = np.array([w[0] for w in weather_data], dtype=np.float64)
        weather_feats = np.array([w[1] for w in weather_data], dtype=np.float32)

        T = len(traffic_timestamps)
        Fw = weather_feats.shape[1]
        weather_aligned = np.zeros((T, Fw), dtype=np.float32)

        for i, t_ts in enumerate(traffic_timestamps):
            # 找到时间差绝对值最小的天气记录（最近邻匹配）
            time_diffs = np.abs(weather_ts - t_ts)
            nearest_idx = int(np.argmin(time_diffs))
            weather_aligned[i] = weather_feats[nearest_idx]

        logger.info(
            f'天气对齐完成: shape={weather_aligned.shape}, '
            f'对齐时间范围 [{min(traffic_timestamps)}, {max(traffic_timestamps)}]'
        )
        return weather_aligned

    # ------------------------------------------------------------------
    # 从原始文件加载并对齐（便捷方法）
    # ------------------------------------------------------------------
    @classmethod
    def from_raw_and_align(cls, weather_raw_path, traffic_raw_path, output_dir):
        """
        从原始 JSON 文件加载天气和交通数据，对齐后保存 weather_aligned.npy

        这是一个便捷的批处理方法，适用于采集完成后的事后对齐。

        Args:
            weather_raw_path: weather_raw.json 路径
            traffic_raw_path: traffic_raw.json 路径
            output_dir: weather_aligned.npy 输出目录

        Returns:
            weather_aligned: (T, 6) numpy 数组
        """
        # 加载天气原始数据
        with open(weather_raw_path, 'r', encoding='utf-8') as f:
            weather_records = json.load(f)
        weather_data = [(r['timestamp'], r['weather']) for r in weather_records]

        # 加载交通时间戳
        with open(traffic_raw_path, 'r', encoding='utf-8') as f:
            traffic_records = json.load(f)
        traffic_ts = [r['timestamp'] for r in traffic_records]

        # 对齐
        weather_aligned = cls.align_with_traffic(weather_data, traffic_ts)

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        aligned_path = os.path.join(output_dir, 'weather_aligned.npy')
        np.save(aligned_path, weather_aligned)
        logger.info(f'保存对齐后的天气数据: {aligned_path}, shape={weather_aligned.shape}')

        return weather_aligned


# ----------------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='和风天气数据采集器 (与gaode_fetcher配合使用, 同地区同时段)'
    )
    parser.add_argument('--days', type=int, default=7,
                        help='采集天数 (默认7天, 与高德采集保持一致)')
    parser.add_argument('--location', type=str, default='101010100',
                        help='和风天气城市ID (默认101010100=北京, '
                             '与高德rectangle北京中关村区域对应)')
    parser.add_argument('--interval', type=int, default=300,
                        help='采集间隔秒数 (默认300s=5min, 与高德采集一致)')
    parser.add_argument('--output_dir', type=str, default='data/gaode',
                        help='数据保存目录 (默认data/gaode, 与高德数据同目录)')
    args = parser.parse_args()

    # 从环境变量读取 API Key
    api_key = os.environ.get('QWEATHER_API_KEY', '')
    if not api_key:
        raise EnvironmentError(
            '请设置环境变量 QWEATHER_API_KEY，例如:\n'
            '  $env:QWEATHER_API_KEY = "your_key"  (PowerShell)\n'
            '  set QWEATHER_API_KEY=your_key       (CMD)'
        )

    # 从环境变量读取专属 API Host (2026年必须设置)
    api_host = os.environ.get('QWEATHER_API_HOST', '')
    if not api_host:
        raise EnvironmentError(
            '和风天气从2026年起需使用专属 API Host，请按以下步骤获取:\n'
            '  1. 登录 https://console.qweather.com/setting\n'
            '  2. 复制你的 API Host (格式如 abc1234xyz.def.qweatherapi.com)\n'
            '  3. 设置环境变量:\n'
            '     $env:QWEATHER_API_HOST = "你的Host"  (PowerShell)'
        )

    fetcher = WeatherFetcher(
        api_key=api_key,
        location=args.location,  # 北京 (与高德rectangle区域对应)
        api_host=api_host,
    )

    # 采集指定天数的天气数据
    duration_hours = args.days * 24
    logger.info(
        f'开始采集天气: {args.days}天={duration_hours}h, '
        f'间隔={args.interval}s, 城市ID={args.location}'
    )
    fetcher.collect_continuous(
        duration_hours=duration_hours,
        output_dir=args.output_dir,
        interval=args.interval,
    )

    # 与交通数据对齐（需要先有交通数据）
    weather_raw = os.path.join(args.output_dir, 'weather_raw.json')
    traffic_raw = os.path.join(args.output_dir, 'traffic_raw.json')
    if os.path.exists(weather_raw) and os.path.exists(traffic_raw):
        WeatherFetcher.from_raw_and_align(weather_raw, traffic_raw, args.output_dir)
    else:
        logger.warning(
            '未找到交通数据，跳过对齐。请先运行 gaode_fetcher.py 采集交通数据。\n'
            '  采集完成后可手动对齐: python -m utils.weather_fetcher --days 0'
        )
