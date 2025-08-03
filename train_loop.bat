@echo off
setlocal

:: =================================================================
:: ==            通用四子棋AI - 自动训练脚本                     ==
:: =================================================================
::
:: 这个脚本会自动循环执行以下两个步骤:
::   1. 运行 self_play.py 来生成训练数据。
::   2. 运行 train.py 来使用新数据训练模型。
::
:: 你可以启动此脚本，让它在后台长时间运行以提升AI性能。
:: 按下 Ctrl+C 可以随时安全地停止训练循环。
::

:: --- 重要配置: Python 解释器路径 ---
:: 如果你的 python 环境已经添加到了系统 PATH 中，可以直接使用 "python"。
:: 但更可靠的做法是指定虚拟环境(venv)中 python.exe 的完整路径。
:: 将下面的 "python" 替换成你的路径, 例如: "C:\path\to\your\project\venv\Scripts\python.exe"

set PYTHON_EXE=python

:: -----------------------------------------------------------------

title ConnectFour AI Training Loop

set "loop_count=0"

:train_loop
    set /a "loop_count+=1"
    cls
    echo.
    echo ================================================================
    echo ==              开始第 %loop_count% 轮 AI 训练循环                 ==
    echo ================================================================
    echo.

    echo.
    echo --- [步骤 1 of 2] ---
    echo --- 正在运行自对弈 (self_play.py) 以生成新的对局数据... ---
    echo.
    %PYTHON_EXE% self_play.py
    
    REM 检查上一步是否成功执行。如果 self_play.py 失败，则退出。
    if %errorlevel% neq 0 (
        echo.
        echo !!! 错误: self_play.py 执行失败，脚本已终止。
        pause
        exit /b %errorlevel%
    )

    echo.
    echo --- [步骤 2 of 2] ---
    echo --- 正在使用新数据训练神经网络 (train.py)... ---
    echo.
    %PYTHON_EXE% train.py

    if %errorlevel% neq 0 (
        echo.
        echo !!! 错误: train.py 执行失败，脚本已终止。
        pause
        exit /b %errorlevel%
    )

    echo.
    echo ================================================================
    echo ==              第 %loop_count% 轮训练完成。准备开始下一轮...            ==
    echo ================================================================
    echo.
    
    REM 短暂暂停5秒，以便查看日志。你可以修改或移除这行。
    timeout /t 5 /nobreak > nul

goto :train_loop