import os

def print_directory_structure(start_path, prefix=''):
    """
    递归打印目录结构
    :param start_path: 起始目录的路径
    :param prefix: 用于缩进的前缀字符串
    """
    # 获取目录下的所有文件和文件夹
    items = os.listdir(start_path)
    items.sort()
    
    for index, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = index == len(items) - 1
        
        # 确定显示的连接符
        connector = '└── ' if is_last else '├── '
        
        # 打印当前项目
        print(f'{prefix}{connector}{item}')
        
        # 如果是目录，则递归处理
        if os.path.isdir(path):
            # 确定下一级的前缀
            extension = '    ' if is_last else '│   '
            print_directory_structure(path, prefix + extension)

# 使用示例
if __name__ == '__main__':
    # 将当前目录作为起始点
    start_path = '.'
    print(f'项目结构:')
    print_directory_structure(start_path)