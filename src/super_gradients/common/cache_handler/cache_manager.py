import os
import pickle
from typing import List, Tuple, Dict, Union, Any, Optional

class CacheObject:
    """
    CacheObject 类封装了单个缓存文件的操作，包括加载、保存和删除缓存。
    """
    def __init__(self, filename: str, cache_path: str):
        """
        初始化缓存对象。
        :param filename: 原始文件名，用于比较修改时间。
        :param cache_path: 缓存文件的存储路径。
        """
        self._filename = filename
        self._cache_path = cache_path
        
    def get_filename(self) -> str:
        """
        获取原始文件名。
        :return: 原始文件名。
        """
        return self._filename
        
    def get_cache_path(self) -> str:
        """
        获取缓存文件路径。
        :return: 缓存文件路径。
        """
        return self._cache_path
    
    def remove_cache(self):
        """
        删除缓存文件，如果存在。
        """
        if os.path.exists(self._cache_path):
            os.remove(self._cache_path)
        
    def save_cache_data(self, new_cache_data: Dict[str, Any]):
        """
        将新的缓存数据写入缓存文件。
        :param new_cache_data: 包含要缓存的新数据的字典。
        """
        with open(self._cache_path, 'wb') as cache_file:
            pickle.dump(new_cache_data, cache_file)

    def load_cache_data(self) -> Dict[str, Any]:
        """
        从缓存文件中加载缓存数据。
        :return: 包含缓存数据的字典，如果文件损坏或过时则返回空字典。
        """
        if os.path.exists(self._cache_path) and os.path.exists(self._filename):
            file_mtime = os.path.getmtime(self._filename)
            cache_mtime = os.path.getmtime(self._cache_path)
            if cache_mtime > file_mtime:
                try:
                    with open(self._cache_path, 'rb') as cache_file:
                        return pickle.load(cache_file)
                except Exception as e:
                    print(f"Error loading cache file {self._cache_path}: {e}")
                    os.remove(self._cache_path)
        return {}
    
    def add_cache_data(self, key: str, data: Any):
        """
        添加或更新缓存文件中的数据。
        :param key: 要更新的数据键。
        :param data: 新的数据值。
        """
        cache_contents = self.load_cache_data()
        cache_contents[key] = data
        self.save_cache_data(cache_contents)
        
class CacheManager:
    """
    CacheManager 类以静态方式管理缓存文件的创建和获取。它配置缓存文件的存储位置和固定的命名规则。
    """
    _base_dir = None  # 默认的基础目录为 None，使用全路径
    _cache_file_suffix = 'cwzcachedpkl'  # 固定的缓存文件后缀

    @staticmethod
    def set_base_dir(base_dir: Optional[str]):
        """
        设置缓存文件的基础目录。
        :param base_dir: 要设置的基础目录，如果为 None 则使用文件的全路径。
        """
        CacheManager._base_dir = base_dir

    @staticmethod
    def _get_cache_path(filename: str) -> str:
        """
        根据原始文件名生成缓存文件的路径。
        :param filename: 原始文件名。
        :return: 缓存文件的完整路径。
        """
        cache_filename = filename + "." + CacheManager._cache_file_suffix
        if CacheManager._base_dir is None:
            # 如果没有设置基础目录，使用全路径且替换目录分隔符
            return cache_filename
        else:
            # 如果设置了基础目录，将文件名与基础目录结合
            cache_filename = cache_filename.replace('\\', '_').replace('/', '_')
            cache_filename = os.path.join(CacheManager._base_dir, cache_filename)
            return cache_filename

    @staticmethod
    def get_cache_obj(filename: str) -> CacheObject:
        """
        获取指定文件的缓存对象。
        :param filename: 原始文件名。
        :return: 对应的缓存对象。
        """
        return CacheObject(filename, CacheManager._get_cache_path(filename))

