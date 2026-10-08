param(
    [string]$Output = (Join-Path $PSScriptRoot '..\config\experiments\sensor_hardware_inventory.json')
)

$serialPorts = @(
    Get-CimInstance Win32_SerialPort -ErrorAction SilentlyContinue |
        Select-Object DeviceID, Name, PNPDeviceID, Status
)

$sensorPattern = 'DHT|BME|SHT|temperature|humidity|light sensor|Arduino|CH340|CP210'
$matchedDevices = @(
    Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Class -eq 'Sensor' -or $_.FriendlyName -match $sensorPattern
        } |
        Select-Object Class, FriendlyName, InstanceId, Status
)

$report = [ordered]@{
    schema_version = 1
    audited_at = (Get-Date).ToString('o')
    computer = $env:COMPUTERNAME
    powershell = $PSVersionTable.PSVersion.ToString()
    serial_ports = $serialPorts
    matched_sensor_or_microcontroller_devices = $matchedDevices
    physical_sensor_available = ($matchedDevices.Count -gt 0)
    integration_verified = $false
    interpretation = @(
        'A generic serial port alone is not evidence of a temperature, humidity, or light sensor.'
        'Integration can be verified only after a device model, wire protocol, calibration, and live readings are available.'
    )
}

$outputPath = [System.IO.Path]::GetFullPath($Output)
$outputDirectory = [System.IO.Path]::GetDirectoryName($outputPath)
[System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $outputPath -Encoding utf8
Write-Output "saved $outputPath"
$report | ConvertTo-Json -Depth 6
