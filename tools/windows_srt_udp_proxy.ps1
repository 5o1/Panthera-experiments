param(
    [int]$ListenPort = 9000,
    [string]$TargetHost = "192.168.30.150",
    [int]$TargetPort = 9000
)

$ErrorActionPreference = "Stop"

$listener = [System.Net.Sockets.UdpClient]::new()
$listener.Client.SetSocketOption(
    [System.Net.Sockets.SocketOptionLevel]::Socket,
    [System.Net.Sockets.SocketOptionName]::ReuseAddress,
    $true
)
$listener.Client.Bind(
    [System.Net.IPEndPoint]::new([System.Net.IPAddress]::Any, $ListenPort)
)

$upstream = [System.Net.Sockets.UdpClient]::new(0)
$upstream.Connect($TargetHost, $TargetPort)
$cameraEndpoint = $null
$cameraReceive = $listener.ReceiveAsync()
$upstreamReceive = $upstream.ReceiveAsync()

Write-Host "SRT UDP proxy listening on 0.0.0.0:$ListenPort -> ${TargetHost}:$TargetPort"

try {
    while ($true) {
        $completed = [System.Threading.Tasks.Task]::WhenAny(
            $cameraReceive,
            $upstreamReceive
        ).GetAwaiter().GetResult()

        if ($completed -eq $cameraReceive) {
            $packet = $cameraReceive.GetAwaiter().GetResult()
            $cameraEndpoint = $packet.RemoteEndPoint
            [void]$upstream.SendAsync(
                $packet.Buffer,
                $packet.Buffer.Length
            ).GetAwaiter().GetResult()
            $cameraReceive = $listener.ReceiveAsync()
        }

        if ($completed -eq $upstreamReceive) {
            $packet = $upstreamReceive.GetAwaiter().GetResult()
            if ($null -ne $cameraEndpoint) {
                [void]$listener.SendAsync(
                    $packet.Buffer,
                    $packet.Buffer.Length,
                    $cameraEndpoint
                ).GetAwaiter().GetResult()
            }
            $upstreamReceive = $upstream.ReceiveAsync()
        }
    }
}
finally {
    $upstream.Dispose()
    $listener.Dispose()
}
