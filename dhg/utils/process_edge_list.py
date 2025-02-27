import json
import os
import pickle
from tqdm import tqdm
import re
from datetime import datetime, timedelta

def service_to_id(service):
    # 服务名称到ID的映射
    service_map = {
        "logservice1": 0,
        "logservice2": 1,
        "mobservice1": 2,
        "mobservice2": 3,
        "redisservice1": 4,
        "redisservice2": 5,
        "dbservice1": 6,
        "dbservice2": 7,
        "webservice1": 8,
        "webservice2": 9
    }
    return service_map.get(service)

def path_to_edge(path):
    # 将路径字符串转换为ID列表
    services = path.split(" -> ")
    return [service_to_id(service) for service in services]

def get_base_hypergraph():
    # 返回基本超图的边列表
    return [
        [8, 5, 2],  # E1
        [0, 9],     # E2
        [7, 1, 5],  # E3
        [3, 6],      # E4
        [0,1],
        [2,3],
        [4,5],
        [6,7],
        [8,9],
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9]
    ]
def get_json_files():
    # 生成日期范围（排除2021-07-19）
    start_date = datetime(2021, 7, 4)
    end_date = datetime(2021, 7, 31)
    date_list = []
    current_date = start_date

    while current_date <= end_date:
        if current_date.day != 19:  # 排除19号
            date_str = current_date.strftime("%Y-%m-%d")
            date_list.append(date_str)
        current_date += timedelta(days=1)

    # 获取所有json文件路径
    json_files = []
    for date in date_list:
        date_path = os.path.join("trace_frequent", date)
        if os.path.exists(date_path):
            for file in os.listdir(date_path):
                if file.startswith("trace_frequent_") and file.endswith(".json"):
                    json_files.append(os.path.join(date_path, file))

    # 按文件名中的数字排序
    json_files.sort(key=lambda x: int(re.search(r'trace_frequent_(\d+)', x).group(1)))
    return json_files
def process_frequent_patterns():
    # 创建edge_list文件夹（如果不存在）
    if not os.path.exists("edge_list"):
        os.makedirs("edge_list")

    # 获取所有trace_frequent文件
    trace_files = get_json_files()

    # 使用tqdm创建进度条
    for file_path in tqdm(trace_files, desc="Processing files"):
        try:
            # 从文件名中提取数字
            file_num = re.search(r'trace_frequent_(\d+)', file_path).group(1)

            # 读取JSON文件
            with open(file_path, 'r') as f:
                data = json.load(f)

            # 获取基本超图边列表
            edge_list = get_base_hypergraph()

            # 从频繁项中添加新边
            added_edges = 0
            for path in data.keys():
                if added_edges >= 30:  # 如果已经添加了20条新边则停止
                    break

                edge = path_to_edge(path)
                if all(x is not None for x in edge):  # 确保所有服务都被正确映射
                    if edge not in edge_list:  # 检查是否重复
                        edge_list.append(edge)
                        added_edges += 1

            # 保存edge_list
            output_path = f"edge_list/edge_list_{file_num}.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(edge_list, f)

            # 计算边的数量
            num_edges = len(edge_list)

            # 打印当前edge_list和边的数量
            print(f"\nEdge list {file_num} created with {num_edges} edges: {edge_list}")

        except Exception as e:
            print(f"\nError processing file {file_path}: {str(e)}")
if __name__ == "__main__":
    process_frequent_patterns()