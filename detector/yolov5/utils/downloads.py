# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Download utils
"""

import os
import subprocess
import urllib
from pathlib import Path

import requests
import torch


def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    file = Path(file)
    assert_msg = (
        f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    )
    try:  # url1
        print(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file))
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        print(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
        os.system(
            f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -"
        )  # curl download, retry and resume on fail
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            print(f"ERROR: {assert_msg}\n{error_msg}")
        print("")


def attempt_download(file, repo="ultralytics/yolov5", release="v6.1"):
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v6.1', etc.
    def github_assets(repository, version="latest"):
        # Return GitHub repo tag (i.e. 'v6.1') and assets (i.e. ['yolov5s.pt', 'yolov5m.pt', ...])
        if version != "latest":
            version = f"tags/{version}"  # i.e. tags/v6.1
        response = requests.get(
            f"https://api.github.com/repos/{repository}/releases/{version}"
        ).json()  # github api
        return response["tag_name"], [
            x["name"] for x in response["assets"]
        ]  # tag, assets

    file = Path(str(file).strip().replace("'", ""))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            file = name.split("?")[
                0
            ]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                print(f"Found {url} locally at {file}")  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1e5)
            return file

        # GitHub assets
        assets = [
            "yolov5n.pt",
            "yolov5s.pt",
            "yolov5m.pt",
            "yolov5l.pt",
            "yolov5x.pt",
            "yolov5n6.pt",
            "yolov5s6.pt",
            "yolov5m6.pt",
            "yolov5l6.pt",
            "yolov5x6.pt",
        ]
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = (
                        subprocess.check_output(
                            "git tag", shell=True, stderr=subprocess.STDOUT
                        )
                        .decode()
                        .split()[-1]
                    )
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        if name in assets:
            url3 = "https://drive.google.com/drive/folders/1EFQTEUeXWSFww0luse2jB9M1QNZQGwNl"  # backup gdrive mirror
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                url2=f"https://storage.googleapis.com/{repo}/{tag}/{name}",  # backup url (optional)
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/{tag} or {url3}",
            )

    return str(file)


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""
