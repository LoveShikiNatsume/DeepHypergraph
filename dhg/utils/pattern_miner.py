# 有点问题
import os
import json
import pandas as pd
from collections import Counter
from tqdm import tqdm
from datetime import datetime

# 频繁项挖掘的最小支持度阈值
MIN_SUPPORT = 0
GLOBAL_FILE_COUNTER = 1  # 全局文件计数器

def load_event_graph(event_graph_file):
    """
    Load event graph from a JSON file.
    """
    try:
        with open(event_graph_file, 'r', encoding='utf-8') as f:
            event_graph = json.load(f)
        return event_graph
    except Exception as e:
        print(f"Error loading {event_graph_file}: {e}")
        return {}


def extract_service_chains(event_graph, max_depth=8):
    """
    提取服务实例链路，使用深度优先搜索，并排除：
    1. 服务自调用的情况
    2. 链路中重复出现的服务
    """
    service_chains = []

    def dfs(node, path, visited_services, depth):
        if depth > max_depth:
            return

        # 提取服务名称
        try:
            service_name = node.split('-')[1].split(' ')[0]
        except:
            service_name = node

        # 如果该服务已经在路径中，则跳过
        if service_name in visited_services:
            return

        # 将当前服务添加到路径和已访问服务集合中
        current_path = path.copy()
        current_path.append(service_name)
        current_visited = visited_services | {service_name}

        # 继续搜索邻接节点
        neighbors = event_graph.get(node, [])
        for neighbor in neighbors:
            try:
                neighbor_service = neighbor.split('-')[1].split(' ')[0]
            except:
                neighbor_service = neighbor

            # 跳过自调用和已访问的服务
            if neighbor_service != service_name and neighbor_service not in visited_services:
                dfs(neighbor, current_path.copy(), current_visited, depth + 1)

        # 如果当前路径包含不同的服务且长度大于1，则保存
        if len(current_path) > 2:
            # 额外检查确保链路中的所有服务都不相同
            if len(set(current_path)) == len(current_path):
                service_chains.append(" -> ".join(current_path))

    # 从每个节点开始深度优先搜索
    for start_node in event_graph:
        dfs(start_node, [], set(), 0)

    return service_chains
def count_frequent_chains(service_chains, min_support=MIN_SUPPORT):
    """
    Count and filter frequent service chains.
    """
    chain_counter = Counter(service_chains)
    frequent_chains = {chain: count for chain, count in chain_counter.items() if count >= min_support}
    return dict(sorted(frequent_chains.items(), key=lambda item: item[1], reverse=True))

def save_frequent_chains(frequent_chains, output_file):
    """
    Save the frequent service chains to a JSON file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(frequent_chains, f, indent=4, ensure_ascii=False)
        print(f"Saved frequent chains to {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")

def process_date_folder(date_folder, trace_json_base_path, trace_frequent_base_path, global_file_counter):
    """
    Process a single date folder to mine frequent service chains.
    """
    global GLOBAL_FILE_COUNTER

    date_path = os.path.join(trace_json_base_path, date_folder)
    if not os.path.exists(date_path):
        print(f"Skipping {date_folder}: Folder does not exist")
        return GLOBAL_FILE_COUNTER

    # 准备输出目录
    output_path = os.path.join(trace_frequent_base_path, date_folder)
    os.makedirs(output_path, exist_ok=True)

    # 获取所有事件图文件
    event_graph_files = [f for f in os.listdir(date_path) if f.startswith("event_graph_") and f.endswith(".json")]

    if not event_graph_files:
        print(f"No event graph files found in {date_folder}")
        return GLOBAL_FILE_COUNTER

    # 处理每个事件图文件
    for event_graph_file in tqdm(event_graph_files, desc=f"Processing {date_folder}", unit="file"):
        event_graph_path = os.path.join(date_path, event_graph_file)
        event_graph = load_event_graph(event_graph_path)

        if not event_graph:
            continue

        # 提取服务链
        service_chains = extract_service_chains(event_graph, max_depth=6)
        print(f"Extracted {len(service_chains)} service chains from {event_graph_file}")

        # 计算频繁项
        frequent_chains = count_frequent_chains(service_chains)
        print(f"Found {len(frequent_chains)} frequent chains in {event_graph_file}")

        # 保存频繁项
        if frequent_chains:  # 确保有输出
            output_file = os.path.join(output_path, f"trace_frequent_{GLOBAL_FILE_COUNTER}.json")
            save_frequent_chains(frequent_chains, output_file)
            GLOBAL_FILE_COUNTER += 1

    return GLOBAL_FILE_COUNTER

def process_all_dates(trace_json_base_path, trace_frequent_base_path, start_date, end_date):
    """
    Process all date folders within the specified range.
    """
    global GLOBAL_FILE_COUNTER

    # 生成日期范围
    date_range = pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d')
    date_folders = [date for date in date_range if date != "2021-07-19"]

    for date_folder in tqdm(date_folders, desc="Processing all dates", unit="date"):
        GLOBAL_FILE_COUNTER = process_date_folder(date_folder, trace_json_base_path, trace_frequent_base_path, GLOBAL_FILE_COUNTER)

if __name__ == '__main__':
    # 定义路径
    trace_json_base_path = "./trace_json"
    trace_frequent_base_path = "./trace_frequent"
    start_date = "2021-07-04"
    end_date = "2021-07-31"

    # 确保输出目录存在
    os.makedirs(trace_frequent_base_path, exist_ok=True)

    # 处理所有日期文件夹
    process_all_dates(trace_json_base_path, trace_frequent_base_path, start_date, end_date)

    print("Processing complete. All frequent chains saved.")