# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Download utils
"""

import logging
import os
import subprocess
import urllib
from pathlib import Path

import requests
import torch


def is_url(url, check=True):
    # 检查一个字符串是否是一个有效的URL，以及这个URL是否存在
    try:
        url = str(url)  # 转为字符串形式
        result = urllib.parse.urlparse(url)  # 解析URL
        # 检查解析出的URL是否有方案和网络位置。如果没有，all函数会返回False，然后assert语句会抛出一个AssertionError异常
        assert all([result.scheme, result.netloc])  # check if is url
        # 尝试打开URL，并检查HTTP响应的状态码是否为200。如果状态码为200，函数返回True；
        # 否则，urllib.request.urlopen函数将抛出一个HTTPError。如果check参数为False，函数直接返回True
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def url_getsize(url='https://ultralytics.com/images/bus.jpg'):
    # 获取一个可下载文件的大小
    response = requests.head(url, allow_redirects=True)  # 发送请求给url，返回一个对象
    # 从响应的头部获取content-length字段，这个字段表示响应体的大小，也就是文件的大小，否则返回-1
    return int(response.headers.get('content-length', -1))


def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(f'Downloading {url} to {file}...')
        # 下载文件，并显示下载进度
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        # 尝试从第二个URL下载文件。调用curl命令，这个命令支持在下载失败时重试，以及在连接中断后恢复下载
        if file.exists():
            file.unlink()  # remove partial downloads
        LOGGER.info(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -# -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # curl download, retry and resume on fail
    finally:
        # 如果文件不存在，或者文件大小小于min_bytes，函数会删除文件，并打印一条错误消息
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
        LOGGER.info('')


def attempt_download(file, repo='ultralytics/yolov5', release='v7.0'):
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v7.0', etc.
    from utils.general import LOGGER

    # 从GitHub的API获取指定仓库的发布版本
    def github_assets(repository, version='latest'):
        # Return GitHub repo tag (i.e. 'v7.0') and assets (i.e. ['yolov5s.pt', 'yolov5m.pt', ...])
        if version != 'latest':
            version = f'tags/{version}'  # i.e. tags/v7.0
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()  # github api
        return response['tag_name'], [x['name'] for x in response['assets']]  # tag, assets

    # 将file转为Path对象
    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():  # 如果文件不存在
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f'Found {url} locally at {file}')  # 文件已经存在
            else:
                safe_download(file=file, url=url, min_bytes=1E5)  # 尝试下载文件
            return file

        # GitHub assets
        assets = [f'yolov5{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '6', '-cls', '-seg')]  # default
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        if name in assets:
            url3 = 'https://drive.google.com/drive/folders/1EFQTEUeXWSFww0luse2jB9M1QNZQGwNl'  # backup gdrive mirror
            safe_download(
                file,
                url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                min_bytes=1E5,
                error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/{tag} or {url3}')
    # 文件存在直接返回
    return str(file)
