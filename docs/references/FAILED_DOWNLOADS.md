# 下载失败列表

检查日期：2026-08-28

## 尚未下载

| 资料 | 域名 | 结果 | 可用替代资料 |
|---|---|---|---|
| OpenCV `VideoCapture` Doxygen HTML | `docs.opencv.org` | 直连和 `127.0.0.1:7897` 均返回 HTTP 403 | 已下载 OpenCV 官方仓库的 [videoio.hpp](source/opencv_videoio.hpp)，包含 `VideoCapture` API 注释 |

需要尝试路由到另一校园 IP 的域名只有：

```text
docs.opencv.org
```

这不是论文下载权限问题。两篇计划所需论文均已从 `arxiv.org` 成功下载。修正规则后可重新运行 `tools/download_references.ps1`；脚本只会重试缺失文件。
