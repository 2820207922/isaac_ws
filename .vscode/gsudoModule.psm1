# Set $gsudoVerbose=$false before importing this module to remove the verbose messages.
if ($null -eq $gsudoVerbose) { $gsudoVerbose = $true; }
# Set $gsudoVerbose=$false before importing this module to remove the gsudo auto-complete functionality.
if ($null -eq $gsudoAutoComplete) { $gsudoAutoComplete = $true; }

$c = @("function Invoke-Gsudo {")
$c += (Get-Content "$PSScriptRoot\Invoke-Gsudo.ps1")
$c += "}"
iex ($c -join "`n" | Out-String)

function gsudo {
    <#
.SYNOPSIS
gsudo is a sudo for windows. It allows to run a command/ScriptBlock with elevated permissions. If no command is specified, it starts an elevated Powershell session.
.DESCRIPTION
# Syntax:
gsudo [options] { ScriptBlock } [ScriptBlock arguments]

gsudo [-n|--new]             # Run command in a new window and dont wait until command exits
      [-w|--wait]            # If --new is specified it wait until it exits.
      [-d 'CMD command']     # To elevate a Win32 CMD command instead of a Powershell script
      [--integrity {i}]      # Run with integrity level [Low, Medium, High, System]
      [-s]                   # Run as `NT AUTHORITY\System` 
      [--ti]                 # Run as Trusted Installer
      [-u|--user {username}] # Run as specific user (prompts for password)
      [--loadProfile]        # Loads the user profile on the elevated Powershell instance before running {ScriptBlock}
      { ScriptBlock }        # Script to elevate
      [-args $argument1[..., $argumentN]] ; # Pass arguments to the ScriptBlock, available as $args[0], $args[1]...

The command to elevate will run in a different process, so it can't access the parent $variables and scope.

More details about gsudo can be found by running: gsudo -h

.EXAMPLE
gsudo { Get-Process }
This run the `Get-Process` command as an administrator.

.EXAMPLE
gsudo { Get-Process $args[0] } -args "WinLogon"
Example case passing parameters to the ScriptBlock.

.INPUTS
You can pipe an input object and will be received as $input in the elevated ScriptBlock.

"WinLogon" | gsudo.exe { Get-Process $input }

.OUTPUTS
The output is determined by the command that is run with gsudo.

.LINK
https://github.com/gerardog/gsudo
#>

    # Note: gsudo is a windows application. 
    # This wrapper only serves the purpose of:
    #  - Adding support for `gsudo !!` on Powershell
    #  - Adding support for `Get-Help gsudo`

    $invocationLine = $MyInvocation.Line -replace "^$($MyInvocation.InvocationName)\s+" # -replace '"','""'

    if ($invocationLine -match "(^| )!!( |$)") { 
        $i = 0;
        do {
            $c = (Get-History | Select-Object -last 1 -skip $i).CommandLine
            $i++;
        } while ($c -eq $MyInvocation.Line -and $c)
        
        if ($c) { 
            if ($gsudoVerbose) { Write-verbose "Elevating Command: '$c'" -Verbose }
            gsudo.exe $c 
        }
        else {
            throw "Failed to find last invoked command in Powershell history."
        }
    }
    elseif ($myinvocation.expectingInput) {
        $input | & gsudo.exe @args 
    } 
    else { 
        & gsudo.exe @args 
    }
}

function Test-IsGsudoCacheAvailable {
    return ('true' -eq (gsudo status CacheAvailable))
}

function Test-IsProcessElevated {
    <#
.Synopsis
    Tests if the user is an administrator *and* the current proces is elevated.
.Description
    Returns true if the current process is elevated.
.Example
    Test-IsAdmin
#>	
    if ($PSVersionTable.Platform -eq 'Unix') {
        return (id -u) -eq 0
    }
    else {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal $identity
        return $principal.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
    }
}

function Test-IsAdminMember {
    <#
.SYNOPSIS
The function Test-IsAdminMember checks if the currently logged-in user is a member of the local administrators group, regardless of the elevation level of the current process.
#>
    $userName = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $adminGroupSid = "S-1-5-32-544"
    $localAdminGroup = Get-LocalGroup -SID $adminGroupSid
    $isAdmin = (Get-LocalGroupMember -Group $localAdminGroup.Name).Where({ $_.Name -eq $userName }).Count -gt 0
    return $isAdmin
}

Function gsudoPrompt {
    $eol = If (Test-IsProcessElevated) { "$([char]27)[1;31m" + ('#') * ($nestedPromptLevel + 1) + "$([char]27)[0m" } else { '>' * ($nestedPromptLevel + 1) };
    "PS $($executionContext.SessionState.Path.CurrentLocation)$eol ";
}

if ($gsudoAutoComplete) {
    #Create an auto-completer for gsudo.

    $verbs = @('status', 'cache', 'config', 'help', '!!')
    $options = @('-d', '--loadProfile', '--system', '--ti', '-k', '--new', '--wait', '--keepShell', '--keepWindow', '--help', '--debug', '--copyNS', '--integrity', '--user')

    $integrityOptions = @("Low", "Medium", "MediumPlus", "High", "System")
    $TrueFalseReset = @('true', 'false', '--reset')

    $suggestions = @{ 
        '--integrity'                 = $integrityOptions;
        '-i'                          = $integrityOptions;
        'cache'                       = @('on', 'off', 'help');
        'config'                      = @('CacheMode', 'CacheDuration', 'LogLevel', 'NewWindow.Force', 'NewWindow.CloseBehaviour', 'Prompt', 'PipedPrompt', 'ForceAttachedConsole', 'ForcePipedConsole', 'ForceVTConsole', 'CopyEnvironmentVariables', 'CopyNetworkShares', 'PowerShellLoadProfile', 'SecurityEnforceUacIsolation', 'ExceptionList');		
        'cachemode'                   = @('Auto', 'Disabled', 'Explicit', '--reset');
        'loglevel'                    = @('All', 'Debug', 'Info', 'Warning', 'Error', 'None', '--reset');
        'NewWindow.CloseBehaviour'    = @('KeepShellOpen', 'PressKeyToClose', 'OsDefault', '--reset');
        'NewWindow.Force'             = $TrueFalseReset;
        'ForceAttachedConsole'        = $TrueFalseReset;
        'ForcePipedConsole'           = $TrueFalseReset;
        'ForceVTConsole'              = $TrueFalseReset;
        'CopyEnvironmentVariables'    = $TrueFalseReset;
        'CopyNetworkShares'           = $TrueFalseReset;
        'PowerShellLoadProfile'       = $TrueFalseReset;
        'SecurityEnforceUacIsolation' = $TrueFalseReset;
		'Status'                      = @('--json', 'CallerPid', 'UserName', 'UserSid', 'IsElevated', 'IsAdminMember', 'IntegrityLevelNumeric', 'IntegrityLevel', 'CacheMode', 'CacheAvailable', 'CacheSessionsCount', 'CacheSessions', 'IsRedirected', '--no-output')
        '--user'                      = @("$env:USERDOMAIN\$env:USERNAME");
        '-u'                          = @("$env:USERDOMAIN\$env:USERNAME")
    }

    $autoCompleter = {
        param($wordToComplete, $commandAst, $cursorPosition)
    
        # gsudo powershell syntax is:
        # gsudo [gsudo options] [optional-gsudo-verb] [gsudo-verb-options | command-to-elevate] [commant-to-elevate-args]
        
        # Will use $phase variable to signal which part of the command is being auto-completed.
        # Phase 1 means autocomplete for [options]
        # Phase 2 means autocomplete for [gsudo-verb]
        # Phase 3 means autocomplete for [verb-options]
        # Phase 4 means [command] is already written.

        $commands = $commandAst.ToString().Substring(0, $cursorPosition - 1).Split(' ') | select -Skip 1;
        if ($wordToComplete) {
            $lastWord = ($commands | select -Last 1 -skip 1)
        }
        else {
            $lastWord = ($commands | select -Last 1)
        }

<# Debugging aids
        # Save the current cursor position
        $originalX = $host.ui.RawUI.CursorPosition.X
        $originalY = $host.ui.RawUI.CursorPosition.Y
        
        # Set the cursor position to (0,0)
        $host.ui.RawUI.CursorPosition = New-Object System.Management.Automation.Host.Coordinates 0, 0
        
        Write-Debug -Debug "wordToComplete = ""$wordToComplete""         "
        Write-Debug -Debug "commandAst = ""$commandAst""         "
        Write-Debug -Debug "cursorPosition = ""$cursorPosition""         "
        Write-Debug -Debug "commands = ""$commands""     ";
        Write-Debug -Debug "lastWord = ""$lastWord""     ";
#>    
        $phase = 1;
    
        foreach ($c in $commands) {
            if ($phase -le 2) {
                if ($verbs -contains $c) { $phase = 3 }
                if ($c -like '{*') { $phase = 4 }
            }
        }

        $filter = "$wordToComplete*"
    
        if ($lastWord -and $suggestions[$lastWord]) {
            $suggestions[$lastWord] -like $filter | % { $_ }
        }
        else {
            if ($phase -lt 3) { 
                if ($wordToComplete -eq '') {
                    # Suggest last 3 executed commands.
                    $lastCommands = Get-History | Select-Object -last 3 | % { "{ $($_.CommandLine) }" }
                
                    if ($lastCommands -is [System.Array]) {
                        # Last one first.
                        $lastCommands[($lastCommands.Length - 1)..0] | % { $_ };
                    }
                    elseif ($lastCommands) {
                        # Only one command.
                        $lastCommands;
                    }
                }
            }
            if ($phase -le 2) { $verbs -like $filter; }	
            if ($phase -le 1) { $options -like $filter; }
            if ($phase -ge 4) { '-args' }

        }
<# Debugging aids
        Write-Debug -Debug "----";

        # Return the cursor position to its original location
        $host.ui.RawUI.CursorPosition = New-Object System.Management.Automation.Host.Coordinates $originalX, $originalY 
#>
    }

    Register-ArgumentCompleter -Native -CommandName 'gsudo' -ScriptBlock $autoCompleter
    Register-ArgumentCompleter -Native -CommandName 'sudo' -ScriptBlock $autoCompleter
}

Export-ModuleMember -function Invoke-Gsudo, gsudo, Test-IsGsudoCacheAvailable, Test-IsProcessElevated, Test-IsAdminMember, gsudoPrompt -Variable gsudoVerbose, gsudoAutoComplete
# SIG # Begin signature block
# MIIr4QYJKoZIhvcNAQcCoIIr0jCCK84CAQExDzANBglghkgBZQMEAgEFADB5Bgor
# BgEEAYI3AgEEoGswaTA0BgorBgEEAYI3AgEeMCYCAwEAAAQQH8w7YFlLCE63JNLG
# KX7zUQIBAAIBAAIBAAIBAAIBADAxMA0GCWCGSAFlAwQCAQUABCDUPXhBSTdC17OD
# DHdMiVBOxTYvFAyhrkzeAADYGfj1ZKCCJQEwggVvMIIEV6ADAgECAhBI/JO0YFWU
# jTanyYqJ1pQWMA0GCSqGSIb3DQEBDAUAMHsxCzAJBgNVBAYTAkdCMRswGQYDVQQI
# DBJHcmVhdGVyIE1hbmNoZXN0ZXIxEDAOBgNVBAcMB1NhbGZvcmQxGjAYBgNVBAoM
# EUNvbW9kbyBDQSBMaW1pdGVkMSEwHwYDVQQDDBhBQUEgQ2VydGlmaWNhdGUgU2Vy
# dmljZXMwHhcNMjEwNTI1MDAwMDAwWhcNMjgxMjMxMjM1OTU5WjBWMQswCQYDVQQG
# EwJHQjEYMBYGA1UEChMPU2VjdGlnbyBMaW1pdGVkMS0wKwYDVQQDEyRTZWN0aWdv
# IFB1YmxpYyBDb2RlIFNpZ25pbmcgUm9vdCBSNDYwggIiMA0GCSqGSIb3DQEBAQUA
# A4ICDwAwggIKAoICAQCN55QSIgQkdC7/FiMCkoq2rjaFrEfUI5ErPtx94jGgUW+s
# hJHjUoq14pbe0IdjJImK/+8Skzt9u7aKvb0Ffyeba2XTpQxpsbxJOZrxbW6q5KCD
# J9qaDStQ6Utbs7hkNqR+Sj2pcaths3OzPAsM79szV+W+NDfjlxtd/R8SPYIDdub7
# P2bSlDFp+m2zNKzBenjcklDyZMeqLQSrw2rq4C+np9xu1+j/2iGrQL+57g2extme
# me/G3h+pDHazJyCh1rr9gOcB0u/rgimVcI3/uxXP/tEPNqIuTzKQdEZrRzUTdwUz
# T2MuuC3hv2WnBGsY2HH6zAjybYmZELGt2z4s5KoYsMYHAXVn3m3pY2MeNn9pib6q
# RT5uWl+PoVvLnTCGMOgDs0DGDQ84zWeoU4j6uDBl+m/H5x2xg3RpPqzEaDux5mcz
# mrYI4IAFSEDu9oJkRqj1c7AGlfJsZZ+/VVscnFcax3hGfHCqlBuCF6yH6bbJDoEc
# QNYWFyn8XJwYK+pF9e+91WdPKF4F7pBMeufG9ND8+s0+MkYTIDaKBOq3qgdGnA2T
# OglmmVhcKaO5DKYwODzQRjY1fJy67sPV+Qp2+n4FG0DKkjXp1XrRtX8ArqmQqsV/
# AZwQsRb8zG4Y3G9i/qZQp7h7uJ0VP/4gDHXIIloTlRmQAOka1cKG8eOO7F/05QID
# AQABo4IBEjCCAQ4wHwYDVR0jBBgwFoAUoBEKIz6W8Qfs4q8p74Klf9AwpLQwHQYD
# VR0OBBYEFDLrkpr/NZZILyhAQnAgNpFcF4XmMA4GA1UdDwEB/wQEAwIBhjAPBgNV
# HRMBAf8EBTADAQH/MBMGA1UdJQQMMAoGCCsGAQUFBwMDMBsGA1UdIAQUMBIwBgYE
# VR0gADAIBgZngQwBBAEwQwYDVR0fBDwwOjA4oDagNIYyaHR0cDovL2NybC5jb21v
# ZG9jYS5jb20vQUFBQ2VydGlmaWNhdGVTZXJ2aWNlcy5jcmwwNAYIKwYBBQUHAQEE
# KDAmMCQGCCsGAQUFBzABhhhodHRwOi8vb2NzcC5jb21vZG9jYS5jb20wDQYJKoZI
# hvcNAQEMBQADggEBABK/oe+LdJqYRLhpRrWrJAoMpIpnuDqBv0WKfVIHqI0fTiGF
# OaNrXi0ghr8QuK55O1PNtPvYRL4G2VxjZ9RAFodEhnIq1jIV9RKDwvnhXRFAZ/ZC
# J3LFI+ICOBpMIOLbAffNRk8monxmwFE2tokCVMf8WPtsAO7+mKYulaEMUykfb9gZ
# pk+e96wJ6l2CxouvgKe9gUhShDHaMuwV5KZMPWw5c9QLhTkg4IUaaOGnSDip0TYl
# d8GNGRbFiExmfS9jzpjoad+sPKhdnckcW67Y8y90z7h+9teDnRGWYpquRRPaf9xH
# +9/DUp/mBlXpnYzyOmJRvOwkDynUWICE5EV7WtgwggWNMIIEdaADAgECAhAOmxiO
# +dAt5+/bUOIIQBhaMA0GCSqGSIb3DQEBDAUAMGUxCzAJBgNVBAYTAlVTMRUwEwYD
# VQQKEwxEaWdpQ2VydCBJbmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5jb20xJDAi
# BgNVBAMTG0RpZ2lDZXJ0IEFzc3VyZWQgSUQgUm9vdCBDQTAeFw0yMjA4MDEwMDAw
# MDBaFw0zMTExMDkyMzU5NTlaMGIxCzAJBgNVBAYTAlVTMRUwEwYDVQQKEwxEaWdp
# Q2VydCBJbmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5jb20xITAfBgNVBAMTGERp
# Z2lDZXJ0IFRydXN0ZWQgUm9vdCBHNDCCAiIwDQYJKoZIhvcNAQEBBQADggIPADCC
# AgoCggIBAL/mkHNo3rvkXUo8MCIwaTPswqclLskhPfKK2FnC4SmnPVirdprNrnsb
# hA3EMB/zG6Q4FutWxpdtHauyefLKEdLkX9YFPFIPUh/GnhWlfr6fqVcWWVVyr2iT
# cMKyunWZanMylNEQRBAu34LzB4TmdDttceItDBvuINXJIB1jKS3O7F5OyJP4IWGb
# NOsFxl7sWxq868nPzaw0QF+xembud8hIqGZXV59UWI4MK7dPpzDZVu7Ke13jrclP
# XuU15zHL2pNe3I6PgNq2kZhAkHnDeMe2scS1ahg4AxCN2NQ3pC4FfYj1gj4QkXCr
# VYJBMtfbBHMqbpEBfCFM1LyuGwN1XXhm2ToxRJozQL8I11pJpMLmqaBn3aQnvKFP
# ObURWBf3JFxGj2T3wWmIdph2PVldQnaHiZdpekjw4KISG2aadMreSx7nDmOu5tTv
# kpI6nj3cAORFJYm2mkQZK37AlLTSYW3rM9nF30sEAMx9HJXDj/chsrIRt7t/8tWM
# cCxBYKqxYxhElRp2Yn72gLD76GSmM9GJB+G9t+ZDpBi4pncB4Q+UDCEdslQpJYls
# 5Q5SUUd0viastkF13nqsX40/ybzTQRESW+UQUOsxxcpyFiIJ33xMdT9j7CFfxCBR
# a2+xq4aLT8LWRV+dIPyhHsXAj6KxfgommfXkaS+YHS312amyHeUbAgMBAAGjggE6
# MIIBNjAPBgNVHRMBAf8EBTADAQH/MB0GA1UdDgQWBBTs1+OC0nFdZEzfLmc/57qY
# rhwPTzAfBgNVHSMEGDAWgBRF66Kv9JLLgjEtUYunpyGd823IDzAOBgNVHQ8BAf8E
# BAMCAYYweQYIKwYBBQUHAQEEbTBrMCQGCCsGAQUFBzABhhhodHRwOi8vb2NzcC5k
# aWdpY2VydC5jb20wQwYIKwYBBQUHMAKGN2h0dHA6Ly9jYWNlcnRzLmRpZ2ljZXJ0
# LmNvbS9EaWdpQ2VydEFzc3VyZWRJRFJvb3RDQS5jcnQwRQYDVR0fBD4wPDA6oDig
# NoY0aHR0cDovL2NybDMuZGlnaWNlcnQuY29tL0RpZ2lDZXJ0QXNzdXJlZElEUm9v
# dENBLmNybDARBgNVHSAECjAIMAYGBFUdIAAwDQYJKoZIhvcNAQEMBQADggEBAHCg
# v0NcVec4X6CjdBs9thbX979XB72arKGHLOyFXqkauyL4hxppVCLtpIh3bb0aFPQT
# SnovLbc47/T/gLn4offyct4kvFIDyE7QKt76LVbP+fT3rDB6mouyXtTP0UNEm0Mh
# 65ZyoUi0mcudT6cGAxN3J0TU53/oWajwvy8LpunyNDzs9wPHh6jSTEAZNUZqaVSw
# uKFWjuyk1T3osdz9HNj0d1pcVIxv76FQPfx2CWiEn2/K2yCNNWAcAgPLILCsWKAO
# QGPFmCLBsln1VWvPJ6tsds5vIy30fnFqI2si/xK4VC0nftg62fC2h5b9W9FcrBjD
# TZ9ztwGpn1eqXijiuZQwggYaMIIEAqADAgECAhBiHW0MUgGeO5B5FSCJIRwKMA0G
# CSqGSIb3DQEBDAUAMFYxCzAJBgNVBAYTAkdCMRgwFgYDVQQKEw9TZWN0aWdvIExp
# bWl0ZWQxLTArBgNVBAMTJFNlY3RpZ28gUHVibGljIENvZGUgU2lnbmluZyBSb290
# IFI0NjAeFw0yMTAzMjIwMDAwMDBaFw0zNjAzMjEyMzU5NTlaMFQxCzAJBgNVBAYT
# AkdCMRgwFgYDVQQKEw9TZWN0aWdvIExpbWl0ZWQxKzApBgNVBAMTIlNlY3RpZ28g
# UHVibGljIENvZGUgU2lnbmluZyBDQSBSMzYwggGiMA0GCSqGSIb3DQEBAQUAA4IB
# jwAwggGKAoIBgQCbK51T+jU/jmAGQ2rAz/V/9shTUxjIztNsfvxYB5UXeWUzCxEe
# AEZGbEN4QMgCsJLZUKhWThj/yPqy0iSZhXkZ6Pg2A2NVDgFigOMYzB2OKhdqfWGV
# oYW3haT29PSTahYkwmMv0b/83nbeECbiMXhSOtbam+/36F09fy1tsB8je/RV0mIk
# 8XL/tfCK6cPuYHE215wzrK0h1SWHTxPbPuYkRdkP05ZwmRmTnAO5/arnY83jeNzh
# P06ShdnRqtZlV59+8yv+KIhE5ILMqgOZYAENHNX9SJDm+qxp4VqpB3MV/h53yl41
# aHU5pledi9lCBbH9JeIkNFICiVHNkRmq4TpxtwfvjsUedyz8rNyfQJy/aOs5b4s+
# ac7IH60B+Ja7TVM+EKv1WuTGwcLmoU3FpOFMbmPj8pz44MPZ1f9+YEQIQty/NQd/
# 2yGgW+ufflcZ/ZE9o1M7a5Jnqf2i2/uMSWymR8r2oQBMdlyh2n5HirY4jKnFH/9g
# Rvd+QOfdRrJZb1sCAwEAAaOCAWQwggFgMB8GA1UdIwQYMBaAFDLrkpr/NZZILyhA
# QnAgNpFcF4XmMB0GA1UdDgQWBBQPKssghyi47G9IritUpimqF6TNDDAOBgNVHQ8B
# Af8EBAMCAYYwEgYDVR0TAQH/BAgwBgEB/wIBADATBgNVHSUEDDAKBggrBgEFBQcD
# AzAbBgNVHSAEFDASMAYGBFUdIAAwCAYGZ4EMAQQBMEsGA1UdHwREMEIwQKA+oDyG
# Omh0dHA6Ly9jcmwuc2VjdGlnby5jb20vU2VjdGlnb1B1YmxpY0NvZGVTaWduaW5n
# Um9vdFI0Ni5jcmwwewYIKwYBBQUHAQEEbzBtMEYGCCsGAQUFBzAChjpodHRwOi8v
# Y3J0LnNlY3RpZ28uY29tL1NlY3RpZ29QdWJsaWNDb2RlU2lnbmluZ1Jvb3RSNDYu
# cDdjMCMGCCsGAQUFBzABhhdodHRwOi8vb2NzcC5zZWN0aWdvLmNvbTANBgkqhkiG
# 9w0BAQwFAAOCAgEABv+C4XdjNm57oRUgmxP/BP6YdURhw1aVcdGRP4Wh60BAscjW
# 4HL9hcpkOTz5jUug2oeunbYAowbFC2AKK+cMcXIBD0ZdOaWTsyNyBBsMLHqafvIh
# rCymlaS98+QpoBCyKppP0OcxYEdU0hpsaqBBIZOtBajjcw5+w/KeFvPYfLF/ldYp
# mlG+vd0xqlqd099iChnyIMvY5HexjO2AmtsbpVn0OhNcWbWDRF/3sBp6fWXhz7Dc
# ML4iTAWS+MVXeNLj1lJziVKEoroGs9Mlizg0bUMbOalOhOfCipnx8CaLZeVme5yE
# Lg09Jlo8BMe80jO37PU8ejfkP9/uPak7VLwELKxAMcJszkyeiaerlphwoKx1uHRz
# NyE6bxuSKcutisqmKL5OTunAvtONEoteSiabkPVSZ2z76mKnzAfZxCl/3dq3dUNw
# 4rg3sTCggkHSRqTqlLMS7gjrhTqBmzu1L90Y1KWN/Y5JKdGvspbOrTfOXyXvmPL6
# E52z1NZJ6ctuMFBQZH3pwWvqURR8AgQdULUvrxjUYbHHj95Ejza63zdrEcxWLDX6
# xWls/GDnVNueKjWUH3fTv1Y8Wdho698YADR7TNx8X8z2Bev6SivBBOHY+uqiirZt
# g0y9ShQoPzmCcn63Syatatvx157YK9hlcPmVoa1oDE5/L9Uo2bC5a4CH2RwwggZj
# MIIEy6ADAgECAhEArSSHt1l2lEEDfGcsJtCFkzANBgkqhkiG9w0BAQwFADBUMQsw
# CQYDVQQGEwJHQjEYMBYGA1UEChMPU2VjdGlnbyBMaW1pdGVkMSswKQYDVQQDEyJT
# ZWN0aWdvIFB1YmxpYyBDb2RlIFNpZ25pbmcgQ0EgUjM2MB4XDTIyMDgzMTAwMDAw
# MFoXDTIzMDgzMTIzNTk1OVowWjELMAkGA1UEBhMCQVIxFTATBgNVBAgMDEJ1ZW5v
# cyBBaXJlczEZMBcGA1UECgwQR2VyYXJkbyBHcmlnbm9saTEZMBcGA1UEAwwQR2Vy
# YXJkbyBHcmlnbm9saTCCAiIwDQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIBALf1
# uQ1SMmuXo3wQciXN92OzFggUJ4rK4unOcm7fNZT1D4N4vI32joOmDjhb5oUQ4DOP
# bE11it65I0R6wp1ilSg2/ItoYlOnR3OEWu56LIuYqZoSYL4SVS81sU8RMq44sQSK
# VJulyjwgK1q2p3FQFqLQoupMgf/kS9Di5V80HZMfGuP/VG9u1hL25+ruuvyHb/4Y
# qQ7h0qTPNEbrKVeydPER2yHWQVGNZFPYrY7P7gbAkkoXqr5byEM35dQpl79SrX4w
# aMB2EtqOjkJlcikT4PCvTt2cLy/LiTMMXznE++YZ933xMMZ2zrxxAXohPOdQPm1t
# SwXPsfvvANyumYRy5poSLIIPu+7o8fWfRM2jQQkrW6v0hzX0UIwRVkvo316H6aS0
# RHp1I8LgzXYB7G6XjYdGGYFLK4vCvNRCK5yDsMskHyJRmBS1nzcCEAE82a2pQP2d
# wSQBFtKyhl7vBbQcNkCkXI+ZAxN3hKFPMAGRFCaJVYAV3g1fYMoAID1b4va//Fjy
# YsmYABcxdM3frCHIuWg8iqrXPNnGX5UYubs3/npi2Mu4K7ZB12yfV3khfOyRiBz9
# q8iYmEG8ZgVeF6SOLpwDQFO0R3VypVhJZEOfDuT+KOEZVRnP2wJURJpe0ZeQ3v0/
# CqY4cUeVYZ03mTWsdhtA1Uw+1rG8Y9E5gJwErWTbAgMBAAGjggGoMIIBpDAfBgNV
# HSMEGDAWgBQPKssghyi47G9IritUpimqF6TNDDAdBgNVHQ4EFgQUFF6RicCkxiAG
# ELk00hac9X9mhSIwDgYDVR0PAQH/BAQDAgeAMAwGA1UdEwEB/wQCMAAwEwYDVR0l
# BAwwCgYIKwYBBQUHAwMwSgYDVR0gBEMwQTA1BgwrBgEEAbIxAQIBAwIwJTAjBggr
# BgEFBQcCARYXaHR0cHM6Ly9zZWN0aWdvLmNvbS9DUFMwCAYGZ4EMAQQBMEkGA1Ud
# HwRCMEAwPqA8oDqGOGh0dHA6Ly9jcmwuc2VjdGlnby5jb20vU2VjdGlnb1B1Ymxp
# Y0NvZGVTaWduaW5nQ0FSMzYuY3JsMHkGCCsGAQUFBwEBBG0wazBEBggrBgEFBQcw
# AoY4aHR0cDovL2NydC5zZWN0aWdvLmNvbS9TZWN0aWdvUHVibGljQ29kZVNpZ25p
# bmdDQVIzNi5jcnQwIwYIKwYBBQUHMAGGF2h0dHA6Ly9vY3NwLnNlY3RpZ28uY29t
# MB0GA1UdEQQWMBSBEmdlcmFyZG9nQGdtYWlsLmNvbTANBgkqhkiG9w0BAQwFAAOC
# AYEAcYE+UdbFGDM9uwzeC5wAcEP9hfPWOkOd0oH0jJsPFe00k1CdEIC71/b7LCQX
# lLNRZHiKlxgfpM+r7td763raFyPyrEYz7/UBYLl2yznseXQgTy9fHY9df/T6z8I9
# X09hyKHRYj0uUhtdWMvMZd+/OVb4IxP2smsB9isJITXLbrXckJQx/pNggQjNREp9
# jKgcL2ejb+Trq+C9J8xlvoIEnvYmZNfHioyjHC5vKC3jMpfXfCGiandFSGNKfp2Y
# n20u4q6WR8hUv18OTAnasMmkAUnLTfPZGkCR/kQTr8wwmQoKwI4jQnVX72C3zhRF
# wsiJfdX+2s+JlW1trdfjTMAkKMoryPYvKR/US380N4tWnqoxVHULcDqFfixjKimn
# Y3tdBWbmsW4gfsHfkHSOBnxsLfX7kAV48yXoNJbfpingwHwc12/372dx5PWsjTV3
# ZzL6zfrs/LfGdX5n7UzzfilSVLcIDV5IZtejI9OQGczLwNHHK5J10S8H9MMqP/HF
# alHJMIIGrjCCBJagAwIBAgIQBzY3tyRUfNhHrP0oZipeWzANBgkqhkiG9w0BAQsF
# ADBiMQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQL
# ExB3d3cuZGlnaWNlcnQuY29tMSEwHwYDVQQDExhEaWdpQ2VydCBUcnVzdGVkIFJv
# b3QgRzQwHhcNMjIwMzIzMDAwMDAwWhcNMzcwMzIyMjM1OTU5WjBjMQswCQYDVQQG
# EwJVUzEXMBUGA1UEChMORGlnaUNlcnQsIEluYy4xOzA5BgNVBAMTMkRpZ2lDZXJ0
# IFRydXN0ZWQgRzQgUlNBNDA5NiBTSEEyNTYgVGltZVN0YW1waW5nIENBMIICIjAN
# BgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAxoY1BkmzwT1ySVFVxyUDxPKRN6mX
# UaHW0oPRnkyibaCwzIP5WvYRoUQVQl+kiPNo+n3znIkLf50fng8zH1ATCyZzlm34
# V6gCff1DtITaEfFzsbPuK4CEiiIY3+vaPcQXf6sZKz5C3GeO6lE98NZW1OcoLevT
# sbV15x8GZY2UKdPZ7Gnf2ZCHRgB720RBidx8ald68Dd5n12sy+iEZLRS8nZH92GD
# Gd1ftFQLIWhuNyG7QKxfst5Kfc71ORJn7w6lY2zkpsUdzTYNXNXmG6jBZHRAp8By
# xbpOH7G1WE15/tePc5OsLDnipUjW8LAxE6lXKZYnLvWHpo9OdhVVJnCYJn+gGkcg
# Q+NDY4B7dW4nJZCYOjgRs/b2nuY7W+yB3iIU2YIqx5K/oN7jPqJz+ucfWmyU8lKV
# EStYdEAoq3NDzt9KoRxrOMUp88qqlnNCaJ+2RrOdOqPVA+C/8KI8ykLcGEh/FDTP
# 0kyr75s9/g64ZCr6dSgkQe1CvwWcZklSUPRR8zZJTYsg0ixXNXkrqPNFYLwjjVj3
# 3GHek/45wPmyMKVM1+mYSlg+0wOI/rOP015LdhJRk8mMDDtbiiKowSYI+RQQEgN9
# XyO7ZONj4KbhPvbCdLI/Hgl27KtdRnXiYKNYCQEoAA6EVO7O6V3IXjASvUaetdN2
# udIOa5kM0jO0zbECAwEAAaOCAV0wggFZMBIGA1UdEwEB/wQIMAYBAf8CAQAwHQYD
# VR0OBBYEFLoW2W1NhS9zKXaaL3WMaiCPnshvMB8GA1UdIwQYMBaAFOzX44LScV1k
# TN8uZz/nupiuHA9PMA4GA1UdDwEB/wQEAwIBhjATBgNVHSUEDDAKBggrBgEFBQcD
# CDB3BggrBgEFBQcBAQRrMGkwJAYIKwYBBQUHMAGGGGh0dHA6Ly9vY3NwLmRpZ2lj
# ZXJ0LmNvbTBBBggrBgEFBQcwAoY1aHR0cDovL2NhY2VydHMuZGlnaWNlcnQuY29t
# L0RpZ2lDZXJ0VHJ1c3RlZFJvb3RHNC5jcnQwQwYDVR0fBDwwOjA4oDagNIYyaHR0
# cDovL2NybDMuZGlnaWNlcnQuY29tL0RpZ2lDZXJ0VHJ1c3RlZFJvb3RHNC5jcmww
# IAYDVR0gBBkwFzAIBgZngQwBBAIwCwYJYIZIAYb9bAcBMA0GCSqGSIb3DQEBCwUA
# A4ICAQB9WY7Ak7ZvmKlEIgF+ZtbYIULhsBguEE0TzzBTzr8Y+8dQXeJLKftwig2q
# KWn8acHPHQfpPmDI2AvlXFvXbYf6hCAlNDFnzbYSlm/EUExiHQwIgqgWvalWzxVz
# jQEiJc6VaT9Hd/tydBTX/6tPiix6q4XNQ1/tYLaqT5Fmniye4Iqs5f2MvGQmh2yS
# vZ180HAKfO+ovHVPulr3qRCyXen/KFSJ8NWKcXZl2szwcqMj+sAngkSumScbqyQe
# JsG33irr9p6xeZmBo1aGqwpFyd/EjaDnmPv7pp1yr8THwcFqcdnGE4AJxLafzYeH
# JLtPo0m5d2aR8XKc6UsCUqc3fpNTrDsdCEkPlM05et3/JWOZJyw9P2un8WbDQc1P
# tkCbISFA0LcTJM3cHXg65J6t5TRxktcma+Q4c6umAU+9Pzt4rUyt+8SVe+0KXzM5
# h0F4ejjpnOHdI/0dKNPH+ejxmF/7K9h+8kaddSweJywm228Vex4Ziza4k9Tm8heZ
# Wcpw8De/mADfIBZPJ/tgZxahZrrdVcA6KYawmKAr7ZVBtzrVFZgxtGIJDwq9gdkT
# /r+k0fNX2bwE+oLeMt8EifAAzV3C+dAjfwAL5HYCJtnwZXZCpimHCUcr5n8apIUP
# /JiW9lVUKx+A+sDyDivl1vupL0QVSucTDh3bNzgaoSv27dZ8/DCCBsIwggSqoAMC
# AQICEAVEr/OUnQg5pr/bP1/lYRYwDQYJKoZIhvcNAQELBQAwYzELMAkGA1UEBhMC
# VVMxFzAVBgNVBAoTDkRpZ2lDZXJ0LCBJbmMuMTswOQYDVQQDEzJEaWdpQ2VydCBU
# cnVzdGVkIEc0IFJTQTQwOTYgU0hBMjU2IFRpbWVTdGFtcGluZyBDQTAeFw0yMzA3
# MTQwMDAwMDBaFw0zNDEwMTMyMzU5NTlaMEgxCzAJBgNVBAYTAlVTMRcwFQYDVQQK
# Ew5EaWdpQ2VydCwgSW5jLjEgMB4GA1UEAxMXRGlnaUNlcnQgVGltZXN0YW1wIDIw
# MjMwggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQCjU0WHHYOOW6w+VLMj
# 4M+f1+XS512hDgncL0ijl3o7Kpxn3GIVWMGpkxGnzaqyat0QKYoeYmNp01icNXG/
# OpfrlFCPHCDqx5o7L5Zm42nnaf5bw9YrIBzBl5S0pVCB8s/LB6YwaMqDQtr8fwkk
# lKSCGtpqutg7yl3eGRiF+0XqDWFsnf5xXsQGmjzwxS55DxtmUuPI1j5f2kPThPXQ
# x/ZILV5FdZZ1/t0QoRuDwbjmUpW1R9d4KTlr4HhZl+NEK0rVlc7vCBfqgmRN/yPj
# yobutKQhZHDr1eWg2mOzLukF7qr2JPUdvJscsrdf3/Dudn0xmWVHVZ1KJC+sK5e+
# n+T9e3M+Mu5SNPvUu+vUoCw0m+PebmQZBzcBkQ8ctVHNqkxmg4hoYru8QRt4GW3k
# 2Q/gWEH72LEs4VGvtK0VBhTqYggT02kefGRNnQ/fztFejKqrUBXJs8q818Q7aESj
# pTtC/XN97t0K/3k0EH6mXApYTAA+hWl1x4Nk1nXNjxJ2VqUk+tfEayG66B80mC86
# 6msBsPf7Kobse1I4qZgJoXGybHGvPrhvltXhEBP+YUcKjP7wtsfVx95sJPC/QoLK
# oHE9nJKTBLRpcCcNT7e1NtHJXwikcKPsCvERLmTgyyIryvEoEyFJUX4GZtM7vvrr
# kTjYUQfKlLfiUKHzOtOKg8tAewIDAQABo4IBizCCAYcwDgYDVR0PAQH/BAQDAgeA
# MAwGA1UdEwEB/wQCMAAwFgYDVR0lAQH/BAwwCgYIKwYBBQUHAwgwIAYDVR0gBBkw
# FzAIBgZngQwBBAIwCwYJYIZIAYb9bAcBMB8GA1UdIwQYMBaAFLoW2W1NhS9zKXaa
# L3WMaiCPnshvMB0GA1UdDgQWBBSltu8T5+/N0GSh1VapZTGj3tXjSTBaBgNVHR8E
# UzBRME+gTaBLhklodHRwOi8vY3JsMy5kaWdpY2VydC5jb20vRGlnaUNlcnRUcnVz
# dGVkRzRSU0E0MDk2U0hBMjU2VGltZVN0YW1waW5nQ0EuY3JsMIGQBggrBgEFBQcB
# AQSBgzCBgDAkBggrBgEFBQcwAYYYaHR0cDovL29jc3AuZGlnaWNlcnQuY29tMFgG
# CCsGAQUFBzAChkxodHRwOi8vY2FjZXJ0cy5kaWdpY2VydC5jb20vRGlnaUNlcnRU
# cnVzdGVkRzRSU0E0MDk2U0hBMjU2VGltZVN0YW1waW5nQ0EuY3J0MA0GCSqGSIb3
# DQEBCwUAA4ICAQCBGtbeoKm1mBe8cI1PijxonNgl/8ss5M3qXSKS7IwiAqm4z4Co
# 2efjxe0mgopxLxjdTrbebNfhYJwr7e09SI64a7p8Xb3CYTdoSXej65CqEtcnhfOO
# HpLawkA4n13IoC4leCWdKgV6hCmYtld5j9smViuw86e9NwzYmHZPVrlSwradOKmB
# 521BXIxp0bkrxMZ7z5z6eOKTGnaiaXXTUOREEr4gDZ6pRND45Ul3CFohxbTPmJUa
# VLq5vMFpGbrPFvKDNzRusEEm3d5al08zjdSNd311RaGlWCZqA0Xe2VC1UIyvVr1M
# xeFGxSjTredDAHDezJieGYkD6tSRN+9NUvPJYCHEVkft2hFLjDLDiOZY4rbbPvlf
# sELWj+MXkdGqwFXjhr+sJyxB0JozSqg21Llyln6XeThIX8rC3D0y33XWNmdaifj2
# p8flTzU8AL2+nCpseQHc2kTmOt44OwdeOVj0fHMxVaCAEcsUDH6uvP6k63llqmjW
# Iso765qCNVcoFstp8jKastLYOrixRoZruhf9xHdsFWyuq69zOuhJRrfVf8y2OMDY
# 7Bz1tqG4QyzfTkx9HmhwwHcK1ALgXGC7KP845VJa1qwXIiNO9OzTF/tQa/8Hdx9x
# l0RBybhG02wyfFgvZ0dl5Rtztpn5aywGRu9BHvDwX+Db2a2QgESvgBBBijGCBjYw
# ggYyAgEBMGkwVDELMAkGA1UEBhMCR0IxGDAWBgNVBAoTD1NlY3RpZ28gTGltaXRl
# ZDErMCkGA1UEAxMiU2VjdGlnbyBQdWJsaWMgQ29kZSBTaWduaW5nIENBIFIzNgIR
# AK0kh7dZdpRBA3xnLCbQhZMwDQYJYIZIAWUDBAIBBQCgfDAQBgorBgEEAYI3AgEM
# MQIwADAZBgkqhkiG9w0BCQMxDAYKKwYBBAGCNwIBBDAcBgorBgEEAYI3AgELMQ4w
# DAYKKwYBBAGCNwIBFTAvBgkqhkiG9w0BCQQxIgQg2+vl1qdtdceR+sYKOEnrVf5k
# 5p3PS5ARrabm2ax5GEIwDQYJKoZIhvcNAQEBBQAEggIAeZkpuAwhN5olTKI1+oFW
# YWTAsltIWr8BHJhS7Utyy/ieu8sEbrgW+YpQ5HMczkPbOElPBFx9JuNSIgB108wM
# S7IHhxzZ7AtgE/AKFWotK/3WFFIw2TJSvY8c3K4mbF9j1o+pD40JM0XSKpbUe9nT
# xQ/ib9hp31P0xbhdm3waYXag8ZoYdfMXalIfB+Qlia/AEHjiUJ6y9Rk6hffo+6IW
# dktEKx85EuUV9wqfMWXu5n9Dsk1dMJCTvx2U3bTt55UnQltYGW7zDqW96SaKR5vi
# QmL8WsPs+nIb/NW/6cFFA8h5rnscrMgXE7N0UTn6QyGg6/0c3i4GOPtC8WgFW1Ro
# z2QRU8WthFY/9FCCZ4EizElRwlUVpDP6zgM7I33pld/1GHOryP4tX5e4Es3P7cAJ
# jQGmWBA8uVpsGNaecsfGyW1liTAUVhoaQP+Uxp00E7RnOwsiy1i6PpRC08SVbNPW
# S/mu4EpGhw5tYcj893O7NL2M3f2PMUS3Fd4CPb5fZii6ZCBiT5o3r5uoSESrpTvZ
# sZqXpBnh3OZoECmpLGvAf/M4Kh+QFrAJaf53kVLJsI5CH4s90VJsEQ4g98RjYtfk
# eHagfxfQHTRHYRGrXsrBhsRtjrawvOLglE1AIcWOVu+46XZBItgk5I6ver19GymP
# nPdP7wgnzL5ruNY6I81vd66hggMgMIIDHAYJKoZIhvcNAQkGMYIDDTCCAwkCAQEw
# dzBjMQswCQYDVQQGEwJVUzEXMBUGA1UEChMORGlnaUNlcnQsIEluYy4xOzA5BgNV
# BAMTMkRpZ2lDZXJ0IFRydXN0ZWQgRzQgUlNBNDA5NiBTSEEyNTYgVGltZVN0YW1w
# aW5nIENBAhAFRK/zlJ0IOaa/2z9f5WEWMA0GCWCGSAFlAwQCAQUAoGkwGAYJKoZI
# hvcNAQkDMQsGCSqGSIb3DQEHATAcBgkqhkiG9w0BCQUxDxcNMjMwODI3MjEzNDM3
# WjAvBgkqhkiG9w0BCQQxIgQg2uyf3sB0RJrLVtsf9FZGLBWq4T7noY2s5Vb3YaKz
# nxowDQYJKoZIhvcNAQEBBQAEggIAImtcX148pDojbfMlVcODKcf35JxhoWR6Mecn
# WLZf2FB9vgLs/yCWtSlVSw+nUc8GCmmQe8qk3aZVnC/Rms3I5Grywj4IptcVlc6L
# ygMag1sU1R/fdhrrqlZnDu222FKEEw+IoZ+ubcN8jAnboYIkgQrmh0RCxdwDtnlI
# cVUjjpRNG4s2cganNX69j5UeoX5eKZMVRhfsiCMBhE763AaD2py8+lye6yn4Qqfe
# 9NAF4sdBNPjPJqcayaN4fqisQXpt2XEt/GJz5QOBk/9JHYeIM1fX2JuUjtMbbnwA
# Ue8iUIVmM09sYgBK2pooLXN7EZNbuCavu4jTMVzLG+xrKjPeHnOqCVUPmG9oGeNn
# qFk5dbBly0OoBAB9OFCB4cb0hfFVGIRAPfHlAOJGl0/bYsQdGnEHuuvU43Zmeri5
# 7nzxbdvbukj1D5dpUt6H9GoxNMfkpTBSiffYn0Je1+Gp7fxDC+tWr2dU+DaI0rEo
# GWeTZXfrW7Mewkv2iTZPEkxmrTM4a2+M9GgxJMKufcG4HJQ9407TgVVzT4b4Dt/X
# FwB9AeAqbeTTne8IHEze1ex3Yj3nJvLyQvDCsah2SsXGit6TFQOgtBz6DhoWUjig
# XCeacCd8zJIysrS3OIwgOQRpb/ihq/2z5pCWaEyo7TVTGkrEGTzdCpx7URAFZfzh
# zH6NILs=
# SIG # End signature block
