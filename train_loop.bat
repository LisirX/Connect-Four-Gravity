@echo off
setlocal enabledelayedexpansion

:: =================================================================
:: ==            通用四子棋AI - 智能训练与评估脚本             ==
:: =================================================================
::
:: 这个脚本会自动循环执行以下步骤:
::   1. 备份当前的最优模型 ("冠军")。
::   2. 运行 self_play.py 生成训练数据。
::   3. 运行 train.py 训练出一个 "挑战者" 模型。
::   4. 运行 arena.py 让挑战者与冠军对战。
::   5. 如果挑战者胜率超过阈值，则将其设为新的冠军。
::      否则，保留原来的冠军，丢弃失败的挑战者。
::
:: 按下 Ctrl+C 可以随时安全地停止训练循环。
::

:: --- 配置区域 ---

:: 1. Python 解释器路径 (如果已在PATH中，可直接用 "python")
set "PYTHON_EXE=python"

:: 2. 模型文件路径
set "CHAMPION_MODEL=models\universal_model.pth"
set "CHAMPION_BACKUP=models\champion_backup.pth"
set "ARCHIVE_DIR=models\archive"

:: 3. 评估阈值 (整数)
:: 新模型的胜率必须达到这个值 (例如 55)，才会被接受为新的冠军。
set "WIN_RATE_THRESHOLD=55"

:: --- 脚本初始化 ---

title ConnectFour AI Smart Training Loop
set "loop_count=0"

:: 检查初始模型是否存在
if not exist "%CHAMPION_MODEL%" (
    echo.
    echo !!! 错误: 找不到初始模型 "%CHAMPION_MODEL%"。
    echo !!! 请先手动运行一次 train.py 来创建一个初始模型。
    pause
    exit /b 1
)

:: 创建存档目录 (如果不存在)
if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%"

:: --- 主循环开始 ---

:main_loop
    set /a "loop_count+=1"
    cls
    echo.
    echo =======================================================================
    echo ==                 开始第 !loop_count! 轮 AI 训练与评估循环                  ==
    echo =======================================================================
    echo.

    echo.
    echo --- [步骤 1/4] 备份当前的冠军模型 ---
    copy /Y "%CHAMPION_MODEL%" "%CHAMPION_BACKUP%"
    if %errorlevel% neq 0 (
        echo !!! 错误: 备份冠军模型失败！
        pause
        exit /b 1
    )
    echo 冠军模型已备份至 "%CHAMPION_BACKUP%"
    echo.
    
    echo.
    echo --- [步骤 2/4] 运行自对弈 (self_play.py) 以生成数据... ---
    echo.
    %PYTHON_EXE% self_play.py
    if %errorlevel% neq 0 (
        echo !!! 错误: self_play.py 执行失败，脚本已终止。
        del "%CHAMPION_BACKUP%"
        pause
        exit /b %errorlevel%
    )

    echo.
    echo --- [步骤 3/4] 训练新的挑战者模型 (train.py)... ---
    echo.
    %PYTHON_EXE% train.py
    if %errorlevel% neq 0 (
        echo !!! 错误: train.py 执行失败，脚本已终止。
        del "%CHAMPION_BACKUP%"
        pause
        exit /b %errorlevel%
    )
    echo 挑战者模型已生成于 "%CHAMPION_MODEL%"

    echo.
    echo --- [步骤 4/4] 运行竞技场，评估挑战者 vs 冠军... ---
    echo.
    %PYTHON_EXE% arena.py --old_model "%CHAMPION_BACKUP%"
    set "WIN_RATE=%errorlevel%"

    echo.
    echo =======================================================================
    echo ==                         评估结果分析                              ==
    echo =======================================================================
    echo.
    echo    挑战者胜率: %WIN_RATE%%%
    echo    晋级阈值:   %WIN_RATE_THRESHOLD%%%
    echo.

    if %WIN_RATE% GEQ %WIN_RATE_THRESHOLD% (
        echo *** 结论: 挑战成功！新模型将被加冕为冠军！ ***
        echo.
        
        :: 生成时间戳用于存档
        for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set "dt=%%I"
        set "YYYY=%dt:~0,4%"
        set "MM=%dt:~4,2%"
        set "DD=%dt:~6,2%"
        set "HH=%dt:~8,2%"
        set "MI=%dt:~10,2%"
        set "SS=%dt:~12,2%"
        set "timestamp=%YYYY%-%MM%-%DD%_%HH%h%MI%m%SS%s"
        
        set "archive_name=model_v!loop_count!_win%WIN_RATE%pct_%timestamp%.pth"
        echo 正在将旧冠军模型存档为: !archive_name!
        move "%CHAMPION_BACKUP%" "%ARCHIVE_DIR%\!archive_name!"
        echo.
        echo 新冠军已确立: "%CHAMPION_MODEL%"
    ) else (
        echo *** 结论: 挑战失败。保留原冠军模型，丢弃挑战者。 ***
        echo.
        echo 正在从备份恢复冠军模型...
        copy /Y "%CHAMPION_BACKUP%" "%CHAMPION_MODEL%"
        del "%CHAMPION_BACKUP%"
        echo 恢复完成。
    )
    
    echo.
    echo =======================================================================
    echo ==           第 !loop_count! 轮循环完成。5秒后将开始下一轮...            ==
    echo =======================================================================
    echo.
    
    timeout /t 5 /nobreak > nul

goto :main_loop