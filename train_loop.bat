@echo off
setlocal enabledelayedexpansion

:: =================================================================
:: ==            ͨ��������AI - ����ѵ���������ű�             ==
:: =================================================================
::
:: ����ű����Զ�ѭ��ִ�����²���:
::   1. ���ݵ�ǰ������ģ�� ("�ھ�")��
::   2. ���� self_play.py ����ѵ�����ݡ�
::   3. ���� train.py ѵ����һ�� "��ս��" ģ�͡�
::   4. ���� arena.py ����ս����ھ���ս��
::   5. �����ս��ʤ�ʳ�����ֵ��������Ϊ�µĹھ���
::      ���򣬱���ԭ���Ĺھ�������ʧ�ܵ���ս�ߡ�
::
:: ���� Ctrl+C ������ʱ��ȫ��ֹͣѵ��ѭ����
::

:: --- �������� ---

:: 1. Python ������·�� (�������PATH�У���ֱ���� "python")
set "PYTHON_EXE=python"

:: 2. ģ���ļ�·��
set "CHAMPION_MODEL=models\universal_model.pth"
set "CHAMPION_BACKUP=models\champion_backup.pth"
set "ARCHIVE_DIR=models\archive"

:: 3. ������ֵ (����)
:: ��ģ�͵�ʤ�ʱ���ﵽ���ֵ (���� 55)���Żᱻ����Ϊ�µĹھ���
set "WIN_RATE_THRESHOLD=55"

:: --- �ű���ʼ�� ---

title ConnectFour AI Smart Training Loop
set "loop_count=0"

:: ����ʼģ���Ƿ����
if not exist "%CHAMPION_MODEL%" (
    echo.
    echo !!! ����: �Ҳ�����ʼģ�� "%CHAMPION_MODEL%"��
    echo !!! �����ֶ�����һ�� train.py ������һ����ʼģ�͡�
    pause
    exit /b 1
)

:: �����浵Ŀ¼ (���������)
if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%"

:: --- ��ѭ����ʼ ---

:main_loop
    set /a "loop_count+=1"
    cls
    echo.
    echo =======================================================================
    echo ==                 ��ʼ�� !loop_count! �� AI ѵ��������ѭ��                  ==
    echo =======================================================================
    echo.

    echo.
    echo --- [���� 1/4] ���ݵ�ǰ�Ĺھ�ģ�� ---
    copy /Y "%CHAMPION_MODEL%" "%CHAMPION_BACKUP%"
    if %errorlevel% neq 0 (
        echo !!! ����: ���ݹھ�ģ��ʧ�ܣ�
        pause
        exit /b 1
    )
    echo �ھ�ģ���ѱ����� "%CHAMPION_BACKUP%"
    echo.
    
    echo.
    echo --- [���� 2/4] �����Զ��� (self_play.py) ����������... ---
    echo.
    %PYTHON_EXE% self_play.py
    if %errorlevel% neq 0 (
        echo !!! ����: self_play.py ִ��ʧ�ܣ��ű�����ֹ��
        del "%CHAMPION_BACKUP%"
        pause
        exit /b %errorlevel%
    )

    echo.
    echo --- [���� 3/4] ѵ���µ���ս��ģ�� (train.py)... ---
    echo.
    %PYTHON_EXE% train.py
    if %errorlevel% neq 0 (
        echo !!! ����: train.py ִ��ʧ�ܣ��ű�����ֹ��
        del "%CHAMPION_BACKUP%"
        pause
        exit /b %errorlevel%
    )
    echo ��ս��ģ���������� "%CHAMPION_MODEL%"

    echo.
    echo --- [���� 4/4] ���о�������������ս�� vs �ھ�... ---
    echo.
    %PYTHON_EXE% arena.py --old_model "%CHAMPION_BACKUP%"
    set "WIN_RATE=%errorlevel%"

    echo.
    echo =======================================================================
    echo ==                         �����������                              ==
    echo =======================================================================
    echo.
    echo    ��ս��ʤ��: %WIN_RATE%%%
    echo    ������ֵ:   %WIN_RATE_THRESHOLD%%%
    echo.

    if %WIN_RATE% GEQ %WIN_RATE_THRESHOLD% (
        echo *** ����: ��ս�ɹ�����ģ�ͽ�������Ϊ�ھ��� ***
        echo.
        
        :: ����ʱ������ڴ浵
        for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set "dt=%%I"
        set "YYYY=%dt:~0,4%"
        set "MM=%dt:~4,2%"
        set "DD=%dt:~6,2%"
        set "HH=%dt:~8,2%"
        set "MI=%dt:~10,2%"
        set "SS=%dt:~12,2%"
        set "timestamp=%YYYY%-%MM%-%DD%_%HH%h%MI%m%SS%s"
        
        set "archive_name=model_v!loop_count!_win%WIN_RATE%pct_%timestamp%.pth"
        echo ���ڽ��ɹھ�ģ�ʹ浵Ϊ: !archive_name!
        move "%CHAMPION_BACKUP%" "%ARCHIVE_DIR%\!archive_name!"
        echo.
        echo �¹ھ���ȷ��: "%CHAMPION_MODEL%"
    ) else (
        echo *** ����: ��սʧ�ܡ�����ԭ�ھ�ģ�ͣ�������ս�ߡ� ***
        echo.
        echo ���ڴӱ��ݻָ��ھ�ģ��...
        copy /Y "%CHAMPION_BACKUP%" "%CHAMPION_MODEL%"
        del "%CHAMPION_BACKUP%"
        echo �ָ���ɡ�
    )
    
    echo.
    echo =======================================================================
    echo ==           �� !loop_count! ��ѭ����ɡ�5��󽫿�ʼ��һ��...            ==
    echo =======================================================================
    echo.
    
    timeout /t 5 /nobreak > nul

goto :main_loop