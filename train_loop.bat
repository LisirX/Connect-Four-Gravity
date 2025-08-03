@echo off
setlocal

:: =================================================================
:: ==            ͨ��������AI - �Զ�ѵ���ű�                     ==
:: =================================================================
::
:: ����ű����Զ�ѭ��ִ��������������:
::   1. ���� self_play.py ������ѵ�����ݡ�
::   2. ���� train.py ��ʹ��������ѵ��ģ�͡�
::
:: ����������˽ű��������ں�̨��ʱ������������AI���ܡ�
:: ���� Ctrl+C ������ʱ��ȫ��ֹͣѵ��ѭ����
::

:: --- ��Ҫ����: Python ������·�� ---
:: ������ python �����Ѿ���ӵ���ϵͳ PATH �У�����ֱ��ʹ�� "python"��
:: �����ɿ���������ָ�����⻷��(venv)�� python.exe ������·����
:: ������� "python" �滻�����·��, ����: "C:\path\to\your\project\venv\Scripts\python.exe"

set PYTHON_EXE=python

:: -----------------------------------------------------------------

title ConnectFour AI Training Loop

set "loop_count=0"

:train_loop
    set /a "loop_count+=1"
    cls
    echo.
    echo ================================================================
    echo ==              ��ʼ�� %loop_count% �� AI ѵ��ѭ��                 ==
    echo ================================================================
    echo.

    echo.
    echo --- [���� 1 of 2] ---
    echo --- ���������Զ��� (self_play.py) �������µĶԾ�����... ---
    echo.
    %PYTHON_EXE% self_play.py
    
    REM �����һ���Ƿ�ɹ�ִ�С���� self_play.py ʧ�ܣ����˳���
    if %errorlevel% neq 0 (
        echo.
        echo !!! ����: self_play.py ִ��ʧ�ܣ��ű�����ֹ��
        pause
        exit /b %errorlevel%
    )

    echo.
    echo --- [���� 2 of 2] ---
    echo --- ����ʹ��������ѵ�������� (train.py)... ---
    echo.
    %PYTHON_EXE% train.py

    if %errorlevel% neq 0 (
        echo.
        echo !!! ����: train.py ִ��ʧ�ܣ��ű�����ֹ��
        pause
        exit /b %errorlevel%
    )

    echo.
    echo ================================================================
    echo ==              �� %loop_count% ��ѵ����ɡ�׼����ʼ��һ��...            ==
    echo ================================================================
    echo.
    
    REM ������ͣ5�룬�Ա�鿴��־��������޸Ļ��Ƴ����С�
    timeout /t 5 /nobreak > nul

goto :train_loop