"""
网络策略控制模块 (data/network_policy.py)
用于在测试环境下强制实施禁网策略 (QUANT_DISABLE_NETWORK=1)，杜绝在测试期间产生任何真实网络外呼。
"""
import os
import socket
from typing import Optional

class NetworkDisabledForTestError(RuntimeError):
    """当系统处于 QUANT_DISABLE_NETWORK=1 测试禁网模式时抛出此异常，防止发出真实网络请求"""
    pass

def is_network_disabled() -> bool:
    """检查是否处于测试禁网模式"""
    return os.environ.get("QUANT_DISABLE_NETWORK") == "1"

def assert_network_allowed(target: Optional[str] = None) -> None:
    """
    检查是否允许发出网络请求。
    如果 QUANT_DISABLE_NETWORK=1，则立即抛出 NetworkDisabledForTestError。
    非测试环境无任何副效应。
    """
    if is_network_disabled():
        target_info = f" (目标: {target})" if target else ""
        raise NetworkDisabledForTestError(
            f"🚨 [测试禁网保护] QUANT_DISABLE_NETWORK=1 已启用，严禁发出真实网络请求{target_info}！"
            "所有测试必须使用本地数据或 mock/fixture。"
        )

_GUARD_INSTALLED = False

def install_network_guard() -> None:
    """
    在底层网络边界安装禁网守卫，拦截 requests、urllib 以及 socket 的真实网络连接。
    只有当 QUANT_DISABLE_NETWORK=1 时才会阻断，非测试模式不受影响。
    """
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return

    orig_connect = socket.socket.connect
    def guarded_connect(self, address):
        assert_network_allowed(f"socket.connect: {address}")
        return orig_connect(self, address)
    socket.socket.connect = guarded_connect

    orig_create_connection = socket.create_connection
    def guarded_create_connection(address, *args, **kwargs):
        assert_network_allowed(f"socket.create_connection: {address}")
        return orig_create_connection(address, *args, **kwargs)
    socket.create_connection = guarded_create_connection

    try:
        import urllib.request
        orig_urlopen = urllib.request.urlopen
        def guarded_urlopen(url, *args, **kwargs):
            full = getattr(url, 'full_url', str(url))
            assert_network_allowed(f"urllib.urlopen: {full}")
            return orig_urlopen(url, *args, **kwargs)
        urllib.request.urlopen = guarded_urlopen
    except Exception:
        pass

    try:
        import requests
        orig_send = requests.Session.send
        def guarded_send(self, request, *args, **kwargs):
            assert_network_allowed(f"requests: {request.url}")
            return orig_send(self, request, *args, **kwargs)
        requests.Session.send = guarded_send
    except Exception:
        pass

    _GUARD_INSTALLED = True
