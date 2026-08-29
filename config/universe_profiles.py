"""
多股票池与策略配置文件管理器 (config/universe_profiles.py)
支持一键切换沪深300核心蓝筹、中证500成长、科技成长龙头、高股息红利等股票池。
"""
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

UNIVERSE_PROFILES: Dict[str, Dict[str, Any]] = {
    "HS300_CORE": {
        "name": "沪深300 核心蓝筹池",
        "description": "代表性 A 股核心资产龙头，流动性充裕，容纳大资金运作",
        "index_code": "000300",
        "symbols": [
            "600519.SH", "000858.SZ", "601318.SH", "300750.SZ", "600036.SH",
            "002594.SZ", "601899.SH", "600900.SH", "000333.SZ", "600276.SH",
            "601166.SH", "002415.SZ", "603288.SH", "600887.SH", "000001.SZ",
            "300059.SZ", "600030.SH", "601012.SH", "002475.SZ", "601988.SH",
            "600048.SH", "000725.SZ", "601668.SH", "600104.SH", "002714.SZ",
            "600309.SH", "300760.SZ", "601288.SH", "601398.SH", "601857.SH"
        ]
    },
    "ZZ500_GROWTH": {
        "name": "中证500 优质成长池",
        "description": "聚焦中盘高弹性、业绩高成长的细分行业冠军标的",
        "index_code": "000905",
        "symbols": [
            "002460.SZ", "300124.SZ", "002049.SZ", "603986.SH", "002241.SZ",
            "300014.SZ", "600438.SH", "002129.SZ", "600584.SH", "002371.SZ",
            "300408.SZ", "603501.SH", "002475.SZ", "300274.SZ", "600745.SH",
            "002601.SZ", "300454.SZ", "600884.SH", "002008.SZ", "603899.SH"
        ]
    },
    "TECH_INNOVATION": {
        "name": "科技创新与先进制造池",
        "description": "半导体芯片、新能源电池、人工智能算力与高端装备龙头",
        "index_code": "000688",
        "symbols": [
            "688981.SH", "300750.SZ", "688012.SH", "688008.SH", "002230.SZ",
            "688111.SH", "300033.SZ", "300308.SZ", "688256.SH", "603501.SH",
            "002415.SZ", "300124.SZ", "603986.SH", "002049.SZ", "688036.SH"
        ]
    },
    "HIGH_DIVIDEND": {
        "name": "高股息红利低波池",
        "description": "银行、公用事业、能源煤炭等高股息率、强现金流与低估值防御标的",
        "index_code": "000015",
        "symbols": [
            "601988.SH", "601398.SH", "601288.SH", "600900.SH", "601088.SH",
            "600028.SH", "601857.SH", "601939.SH", "600011.SH", "600036.SH",
            "601006.SH", "601668.SH", "600019.SH", "600938.SH", "601225.SH"
        ]
    }
}


class UniverseProfileManager:
    """股票池配置管理器"""

    @classmethod
    def list_profiles(cls) -> List[str]:
        return list(UNIVERSE_PROFILES.keys())

    @classmethod
    def get_profile_info(cls, profile_key: str) -> Dict[str, Any]:
        return UNIVERSE_PROFILES.get(profile_key, UNIVERSE_PROFILES["HS300_CORE"])

    @classmethod
    def get_symbols(cls, profile_key: str) -> List[str]:
        info = cls.get_profile_info(profile_key)
        return info["symbols"]
