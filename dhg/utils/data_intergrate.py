
import os
import json
import pandas as pd
import pytz
from tqdm import tqdm
from datetime import datetime, timedelta


def load_groundtruth(groundtruth_file):
    """
    加载 groundtruth 数据，用于定义时间片段
    """
    groundtruth = pd.read_csv(groundtruth_file)

    # 检查是否存在 'st_time' 列
    if 'st_time' not in groundtruth.columns:
        raise ValueError("groundtruth.csv 中没有 'st_time' 列")

    time_slices = []

    for _, row in groundtruth.iterrows():
        try:
            # 按示例格式解析 st_time: 2021-07-04 00:37:11.553000
            start_time = datetime.strptime(row['st_time'], '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            print(f"Error parsing time for row: {row['st_time']}. Skipping this row.")
            continue

        # 固定时间段为 600 秒
        duration = timedelta(seconds=600)
        end_time = start_time + duration

        time_slices.append((start_time, end_time))

    print(f"Loaded {len(time_slices)} time slices: {time_slices[:5]}")  # 打印前 5 个时间片
    return time_slices

def get_events_within_time_slice(trace_reader, start_time, end_time):
    """
    从 trace 数据中筛选出在特定时间片段内的事件
    """
    # 创建上海时区对象
    shanghai_tz = pytz.timezone('Asia/Shanghai')

    # 将 start_time 和 end_time 转换为带时区的 Timestamp
    start_time = pd.Timestamp(start_time).tz_localize(shanghai_tz)
    end_time = pd.Timestamp(end_time).tz_localize(shanghai_tz)

    # 将 Unix 时间戳转换为 datetime
    trace_reader['st_time'] = pd.to_datetime(trace_reader['st_time'], unit='s')

    # 只在时间还没有时区信息时才添加时区
    try:
        trace_reader['st_time'] = trace_reader['st_time'].dt.tz_convert(shanghai_tz)
    except TypeError:
        trace_reader['st_time'] = trace_reader['st_time'].dt.tz_localize('UTC').dt.tz_convert(shanghai_tz)

    print(f"Trace st_time type: {trace_reader['st_time'].dtype}")  # 检查 st_time 类型

    filtered_events = trace_reader[(trace_reader['st_time'] >= start_time) & (trace_reader['st_time'] < end_time)]

    print(f"Filtered {len(filtered_events)} events for time slice {start_time} - {end_time}")
    return filtered_events


def generate_event_graph(events, log_template_miner=None):
    """
    Generate event graph from filtered events, including both start and end events for all spans.

    Args:
        events: DataFrame containing filtered trace events.
        log_template_miner: Optional log template miner (can be None if not needed)

    Returns:
        Event graph with structured relationships between start and end events
    """
    event_graph = {}

    # 先为所有span创建start和end节点以及它们之间的基本连接
    for span_id, span_events in events.groupby('span_id'):
        service_name = span_events.iloc[0]['service_name']
        start_key = f"{span_id}-{service_name} start"
        end_key = f"{span_id}-{service_name} end"

        # 添加start到end的连接
        if start_key not in event_graph:
            event_graph[start_key] = []
        event_graph[start_key].append(end_key)

    # 处理父子span关系
    for span_id, span_events in events.groupby('span_id'):
        parent_id = span_events.iloc[0].get('parent_id')

        if parent_id and parent_id != 'root':
            # 找到父span的事件
            parent_events = events[events['span_id'] == parent_id]

            if not parent_events.empty:
                # 获取父子span的服务名
                parent_service = parent_events.iloc[0]['service_name']
                child_service = span_events.iloc[0]['service_name']

                # 创建父子span的key
                parent_start_key = f"{parent_id}-{parent_service} start"
                child_start_key = f"{span_id}-{child_service} start"
                child_end_key = f"{span_id}-{child_service} end"

                # 从父span的start连接到子span的start
                if parent_start_key not in event_graph:
                    event_graph[parent_start_key] = []
                if child_start_key not in event_graph[parent_start_key]:
                    event_graph[parent_start_key].append(child_start_key)

                # 确保子span的start连接到它自己的end
                if child_start_key not in event_graph:
                    event_graph[child_start_key] = []
                if child_end_key not in event_graph[child_start_key]:
                    event_graph[child_start_key].append(child_end_key)

    return event_graph

def process_date_folder(date_folder, base_path, output_base_path):
    """
    Process a single date folder to generate event graphs for each time slice.
    Args:
        date_folder: Name of the date folder (e.g., "2021-07-04").
        base_path: Root path containing the date folders.
        output_base_path: Path to save the resulting JSON files.
    """
    date_path = os.path.join(base_path, date_folder)
    trace_file = os.path.join(date_path, "trace", "trace.csv")
    groundtruth_file = os.path.join(date_path, "groundtruth.csv")

    if not os.path.exists(trace_file) or not os.path.exists(groundtruth_file):
        print(f"Skipping {date_folder}: Missing trace.csv or groundtruth.csv")
        return
    print(f"Trying to read trace file from: {trace_file}")

    # Load trace and groundtruth data
    print(f"Processing {date_folder}...")
    trace_reader = pd.read_csv(trace_file)
    print(f"Trace file loaded successfully with {len(trace_reader)} rows.")
    print(f"Trace st_time range: {trace_reader['st_time'].min()} to {trace_reader['st_time'].max()}")
    print(trace_reader.head())

    time_slices = load_groundtruth(groundtruth_file)
    if not time_slices:
        print(f"No time slices loaded from {groundtruth_file}")


    # Prepare output directory
    output_path = os.path.join(output_base_path, date_folder)
    os.makedirs(output_path, exist_ok=True)

    # Process each time slice
    for index, (start_time, end_time) in enumerate(tqdm(time_slices, desc=f"Processing slices for {date_folder}")):
        # Filter events within the time slice
        events = get_events_within_time_slice(trace_reader, start_time, end_time)
        print(f"Time slice {index}: Found {len(events)} events")
        # Generate event graph for the time slice
        event_graph = generate_event_graph(events)
        print(f"Generated event graph for slice {index}: {event_graph}")

# Save event graph to JSON file
        output_file = os.path.join(output_path, f"event_graph_{index}.json")
        print(f"Saving event graph to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(event_graph, f, indent=4)

def process_all_dates(base_path, output_base_path, start_date, end_date):
    """
    Process all date folders within the specified range.
    Args:
        base_path: Root path containing the date folders.
        output_base_path: Path to save the resulting JSON files.
        start_date: Start date (e.g., "2021-07-04").
        end_date: End date (e.g., "2021-07-31").
    """
    # Generate a list of all date folders
    date_folders = [os.path.join(base_path, date) for date in pd.date_range(start_date, end_date).strftime('%Y-%m-%d')]

    for date_folder in date_folders:
        # Extract folder name for processing
        folder_name = os.path.basename(date_folder)
        if os.path.exists(date_folder):
            process_date_folder(folder_name, base_path, output_base_path)
        else:
            print(f"Skipping {folder_name}: Folder does not exist")


if __name__ == '__main__':
    base_path = "./new_gaia"
    output_base_path = "./trace_json"
    start_date = "2021-07-04"
    end_date = "2021-07-31"

    # Process all date folders in the specified range
    process_all_dates(base_path, output_base_path, start_date, end_date)

    print("Processing complete. All event graphs saved.")