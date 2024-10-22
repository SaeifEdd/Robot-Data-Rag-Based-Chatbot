import pandas as pd
import numpy as np
import json
import re
from datetime import datetime

#trnasform date
def format_date(date_string):
    date_obj = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    return date_obj.strftime("%Y-%m-%d %H:%M:%S")


def first_transform(data):
    result = []

    # Transform main data
    source = data[0]['_source']
    result.append(f"Device ID: {source['id']}, Reference: {source['reference']}")
    result.append(f"Connection State: {'Connected' if source['connectionState'] else 'Disconnected'}")
    result.append(f"Patrol State: {source['patrolState']}, Status: {source['patrolStatus']}")
    result.append(f"Guidance State: {source['guidanceState']}")
    result.append(
        f"Battery: {source['battery']}%, RTK Correction: {'Active' if source['rtkCorrection'] else 'Inactive'}")
    result.append(f"Connection to Station: {'Yes' if source['connectionStation'] else 'No'}")
    result.append(f"Emergency Stop: {'Activated' if source['emergencyStop'] else 'Not Activated'}")
    result.append(f"Light: {'On' if source['light'] else 'Off'}, Charging: {'Yes' if source['charging'] else 'No'}")
    result.append(f"Speed: {source['speed']} km/h, Distance Travelled: {source['distanceTravelled']} km")
    result.append(f"Optical Zoom: {source['opticalZoom']}x, Dome Orientation: {source['domeOrientation']}°")
    result.append(f"Work Rounds: {source['workRounds']}, Work Hours: {source['workHours']}")

    # Handle missions, buttonActionHandlers, detectionEvents, and missionLogs
    result.append(f"Missions: {len(source['missions'])} mission(s)")
    result.append(f"Button Action Handlers: {len(source['buttonActionHandlers'])} handler(s)")
    result.append(f"Detection Events: {len(source['detectionEvents'])} event(s)")
    result.append(f"Mission Logs: {len(source['missionLogs'])} log(s)")

    return result


def error_transform(data):
    result = []
    source = data[0]['_source']
    if 'errorLogs' in source:
        error_logs = source['errorLogs']

        # Handle ipStatus
        if 'ipStatus' in error_logs:
            for item in error_logs['ipStatus']:
                date = format_date(item['date'])
                status = "not working" if item['Status'] == 'False' else "working"
                ip_info = ', '.join([f"{k}: {v}" for d in item['ip'] for k, v in d.items()])
                result.append(f"IP Status Error on {date}: {item['info']} ({status}). IPs: {ip_info}")

        # Handle gnssStatuses
        if 'gnssStatuses' in error_logs:
            for item in error_logs['gnssStatuses']:
                date = format_date(item['date'])
                result.append(f"GNSS Status on {date}: {item['INFO']}. Satellites: {item['numSV']}, "
                              f"Fix Type: {item['fixType']}, Diff Mode: {item['diffMode']}, "
                              f"Diff Source: {item['diffSource']}, Carrier Status: {item['carrierStatus']}")

        # Handle Lidar
        if 'Lidar' in error_logs:
            for item in error_logs['Lidar']:
                date = format_date(item['date'])
                status = "not working" if item['Status'] == 'False' else "working"
                result.append(f"Lidar Status on {date}: {item['info']} ({status}). IP: {item['ip']}")

        return result


def warning_transform(data):
    source = data[0]['_source']
    warning_logs = []
    if 'warningLogs' in source:
        warn_logs = source['warningLogs']
        # Handle ipStatus
        if 'ipStatus' in warn_logs:
            for item in warn_logs['ipStatus']:
                date = format_date(item['date'])
                status = "not working" if item['Status'] == 'False' else "working"
                warning_logs.append(f"IP Warning on {date}: {item['info']} ({status}). IP: {item['ip']}")

        # Handle gnssStatuses
        if 'gnssStatuses' in warn_logs:
            for item in warn_logs['gnssStatuses']:
                date = format_date(item['date'])
                warning_logs.append(f"GNSS Status on {date}: {item['INFO']}. Satellites: {item['numSV']}, "
                                    f"Fix Type: {item['fixType']}, Diff Mode: {item['diffMode']}, "
                                    f"Diff Source: {item['diffSource']}, Carrier Status: {item['carrierStatus']}")

    return warning_logs


def connection_transform(logs):
    source = data[0]['_source']
    connection_status = []

    # Handle connection statuses
    if 'connection' in source:
        for item in source['connection']:
            date = format_date(item['date'])
            status = "Connected" if item['value'] else "Disconnected"
            connection_status.append(f"Connection Status on {date}: {status}")

    return connection_status

if __name__ == "__main__":
    # Open and read the JSON file
    with open('cleaned_output.json', 'r') as file:
        data = json.load(file)

    transformed_data = []
    # basic information
    basics = first_transform(data)
    transformed_data.append(basics)
    # Transform error logs
    error_logs_output = error_transform(data)
    transformed_data.append(error_logs_output)
    # Transform warning logs
    warning_logs_output = warning_transform(data)
    transformed_data.append(warning_logs_output)
    # Transform connection status logs
    connection_status_output = connection_transform(data)
    transformed_data.append(connection_status_output)
    # dump it in a json
    with open('transformed_data.json', 'w') as outfile:
        json.dump(transformed_data, outfile)