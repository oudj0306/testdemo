# 切换到本机 MySQL

优化版默认连接本机 MySQL 8。请准备一个专用空数据库和可在其中建表、读写的账号。已有 SQLite 数据时，先迁移，再启动应用。保留原来的 `data` 目录，答卷和裁剪图仍从这里读取。

在优化版目录运行以下 PowerShell 命令，将账号和库名改为本机实际值。密码由终端交互输入。

```powershell
$env:GRADING_DB_BACKEND = 'mysql'
$env:GRADING_MYSQL_DATABASE = 'grading_agent'
$env:GRADING_MYSQL_USER = 'grading_agent'
$env:GRADING_MYSQL_PORT = '3306'
$secret = Read-Host 'MySQL password' -AsSecureString
$env:GRADING_MYSQL_PASSWORD = [System.Net.NetworkCredential]::new('', $secret).Password
$env:PYTHONPATH = (Resolve-Path .\src).Path
$python = 'D:\ou\agent\.venv-gpu\Scripts\python.exe'
& $python -m pip install 'PyMySQL[rsa]>=1.1,<2'
```

暂停批改并关闭旧进程，确保切换期间 SQLite 没有新写入。把原库复制到 MySQL，然后启动。

```powershell
& $python -m grading_agent.storage.migrate .\data\grading.db
if ($LASTEXITCODE -ne 0) { throw '迁移未成功，请检查报错' }
.\start-optimization.ps1 -Port 8001
```

迁移工具逐表校验记录数，在同一事务内提交业务数据。队列里的上传二进制、历史批改及复核记录也会复制。已存在业务记录的目标库会被拒绝。建表属于 MySQL 独立提交的操作，失败时目标库可能留下空表。评分缓存会按需重建。空白新项目可以直接启动，由应用建表。

如果要切回旧库，在停止新服务后，将 `GRADING_DB_BACKEND` 设为 `sqlite` 再启动。切换后的 MySQL 新写入不会同步回旧库。

应用使用一个服务进程运行队列。若有大答卷，确保本机 MySQL 的 `max_allowed_packet` 能容纳上传数据。
