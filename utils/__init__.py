"""工具库"""
from .dataloader import PEMS04Loader, create_data_loaders
from .metrics import masked_mae, masked_rmse, masked_smape, compute_all_metrics
from .early_stopping import EarlyStopping
from .topo_visualizer import (plot_adjacency_heatmap, plot_beta_trajectory,
                              plot_topo_comparison)

# 高德数据采集与加载（阶段三新增）
# 这些模块依赖 requests 库，仅在采集场景需要，故延迟导入以避免影响纯训练流程
def __getattr__(name):
    """延迟导入高德相关模块，避免无 requests 时影响训练"""
    if name == 'GaodeTrafficFetcher':
        from .gaode_fetcher import GaodeTrafficFetcher
        return GaodeTrafficFetcher
    if name == 'WeatherFetcher':
        from .weather_fetcher import WeatherFetcher
        return WeatherFetcher
    if name == 'create_gaode_data_loaders':
        from .gaode_loader import create_gaode_data_loaders
        return create_gaode_data_loaders
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
