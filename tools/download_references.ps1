param(
    [string]$OutputRoot = ".\docs\references",
    [string]$ProxyUrl = "http://127.0.0.1:7897"
)

$ErrorActionPreference = "Stop"

$items = @(
    @{ Kind = "paper"; File = "papers/BlazePose_2006.10204.pdf"; Url = "https://arxiv.org/pdf/2006.10204" },
    @{ Kind = "paper"; File = "papers/MediaPipe_Hands_2006.10214.pdf"; Url = "https://arxiv.org/pdf/2006.10214" },
    @{ Kind = "mediapipe"; File = "docs/mediapipe_pose_landmarker_python.html"; Url = "https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python" },
    @{ Kind = "mediapipe"; File = "docs/mediapipe_pose_landmarker_overview.html"; Url = "https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker" },
    @{ Kind = "mediapipe"; File = "docs/mediapipe_gesture_recognizer_overview.html"; Url = "https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer" },
    @{ Kind = "mediapipe"; File = "docs/mediapipe_gesture_recognizer_python.html"; Url = "https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python" },
    @{ Kind = "mediapipe"; File = "docs/mediapipe_custom_gesture_recognizer.html"; Url = "https://developers.google.com/edge/mediapipe/solutions/customization/gesture_recognizer" },
    @{ Kind = "mediapipe"; File = "docs/mediapipe_python_setup.html"; Url = "https://developers.google.com/edge/mediapipe/solutions/setup_python" },
    @{ Kind = "mediapipe"; File = "docs/mediapipe_holistic_blog.html"; Url = "https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/" },
    @{ Kind = "model"; File = "models/pose_landmarker_full.task"; Url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" },
    @{ Kind = "model"; File = "models/gesture_recognizer.task"; Url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task" },
    @{ Kind = "ros2"; File = "docs/ros2_humble_install_ubuntu.html"; Url = "https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html" },
    @{ Kind = "ros2"; File = "docs/ros2_humble_tutorials.html"; Url = "https://docs.ros.org/en/humble/Tutorials.html" },
    @{ Kind = "ros2"; File = "docs/ros2_python_pub_sub.html"; Url = "https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html" },
    @{ Kind = "ros2"; File = "docs/ros2_create_package.html"; Url = "https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html" },
    @{ Kind = "ros2"; File = "docs/ros2_python_parameters.html"; Url = "https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-Python.html" },
    @{ Kind = "ros2"; File = "docs/ros2_rosbag2.html"; Url = "https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html" },
    @{ Kind = "robot"; File = "source/Panthera-HT_ROS2-humble.zip"; Url = "https://github.com/HighTorque-Robotics/Panthera-HT_ROS2/archive/refs/heads/humble.zip" },
    @{ Kind = "math"; File = "docs/scipy_rotation.html"; Url = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html" },
    @{ Kind = "math"; File = "docs/scipy_slerp.html"; Url = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html" },
    @{ Kind = "camera"; File = "docs/opencv_videocapture.html"; Url = "https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html" },
    @{ Kind = "camera-fallback"; File = "source/opencv_videoio.hpp"; Url = "https://raw.githubusercontent.com/opencv/opencv/4.x/modules/videoio/include/opencv2/videoio.hpp" },
    @{ Kind = "protocol"; File = "standards/RFC6455_WebSocket.txt"; Url = "https://www.rfc-editor.org/rfc/rfc6455.txt" },
    @{ Kind = "protocol"; File = "docs/python_websockets.html"; Url = "https://websockets.readthedocs.io/" },
    @{ Kind = "wsl"; File = "docs/microsoft_wsl_install.html"; Url = "https://learn.microsoft.com/windows/wsl/install" },
    @{ Kind = "wsl"; File = "docs/microsoft_wsl_usb.html"; Url = "https://learn.microsoft.com/en-us/windows/wsl/connect-usb" },
    @{ Kind = "tool"; File = "docs/git_homepage.html"; Url = "https://git-scm.com/" }
)

$resolvedRoot = [System.IO.Path]::GetFullPath($OutputRoot)
[System.IO.Directory]::CreateDirectory($resolvedRoot) | Out-Null

function Invoke-CurlDownload {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$Proxy
    )

    $partial = "$Destination.partial"
    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }

    $arguments = @(
        "--location",
        "--fail-with-body",
        "--silent",
        "--show-error",
        "--retry", "2",
        "--retry-delay", "1",
        "--connect-timeout", "15",
        "--max-time", "120",
        "--user-agent", "Panthera-HT-reference-downloader/1.0",
        "--output", $partial
    )
    if ($Proxy) {
        $arguments += @("--proxy", $Proxy)
    }
    $arguments += $Url

    & curl.exe @arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0 -and (Test-Path -LiteralPath $partial) -and ((Get-Item -LiteralPath $partial).Length -gt 0)) {
        Move-Item -LiteralPath $partial -Destination $Destination -Force
        return $true
    }
    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }
    return $false
}

$results = foreach ($item in $items) {
    $destination = Join-Path $resolvedRoot $item.File
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($destination)) | Out-Null

    if ((Test-Path -LiteralPath $destination) -and ((Get-Item -LiteralPath $destination).Length -gt 0)) {
        $route = "existing"
        $ok = $true
    }
    else {
        $route = "direct"
        $ok = Invoke-CurlDownload -Url $item.Url -Destination $destination -Proxy ""
        if (-not $ok) {
            $route = "proxy"
            $ok = Invoke-CurlDownload -Url $item.Url -Destination $destination -Proxy $ProxyUrl
        }
    }

    if ($ok) {
        $downloaded = Get-Item -LiteralPath $destination
        $hash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
        $status = "downloaded"
        $bytes = $downloaded.Length
    }
    else {
        $hash = ""
        $status = "failed"
        $bytes = 0
    }

    [pscustomobject]@{
        status = $status
        route = $route
        kind = $item.Kind
        file = $item.File
        bytes = $bytes
        sha256 = $hash
        domain = ([uri]$item.Url).Host
        url = $item.Url
    }
}

$logPath = Join-Path $resolvedRoot "download-log.csv"
$results | Export-Csv -LiteralPath $logPath -NoTypeInformation -Encoding utf8
$results | Format-Table status, route, kind, bytes, domain, file -AutoSize

$failed = @($results | Where-Object status -eq "failed")
Write-Output "TOTAL=$($results.Count) DOWNLOADED=$($results.Count - $failed.Count) FAILED=$($failed.Count)"
if ($failed.Count -gt 0) {
    Write-Output "FAILED_DOMAINS=$((($failed.domain | Sort-Object -Unique) -join ','))"
    exit 2
}
