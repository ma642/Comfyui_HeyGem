import imageio
import numpy as np
import torch
import subprocess
import os
import sys
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCKER_COMPOSE_FILE = os.path.join(BASE_DIR, "docker-compose-lite.yml")
CONTAINER_NAME = "heygem-gen-video"
VOLUME_ENV_VAR = "HEGEM_FACE2FACE_DATA_PATH"

os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1' # 有些库可能只认小写

def _run_docker_command(command_parts, check_result=False, capture_output=True, text_output=True, env=None):
    """
    内部辅助函数，用于执行 Docker 或 Docker Compose 命令。
    """
    try:
        result = subprocess.run(
            command_parts,
            capture_output=capture_output,
            text=text_output,
            check=check_result,
            env=env
        )
        return result
    except FileNotFoundError:
        print(f"Error: '{command_parts[0]}' command not found.", file=sys.stderr)
        print(f"Please ensure Docker or Docker Compose is installed and in your system's PATH.", file=sys.stderr)
        raise # 重新抛出异常，让调用者处理
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(e.cmd)}", file=sys.stderr)
        print(f"Exit Code: {e.returncode}", file=sys.stderr)
        if e.stdout: print(f"Stdout:\n{e.stdout}", file=sys.stderr)
        if e.stderr: print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        raise # 重新抛出异常
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        raise # 重新抛出异常

def is_docker_container_running(container_name):
    """
    检查指定名称的 Docker 容器是否正在运行。
    这个检查是全局性的，不局限于某个 compose 文件。
    """
    print(f"Checking if container '{container_name}' is currently running...")
    command = [
        "docker", "ps",
        "-q",  # 只输出容器ID
        "--filter", f"name={container_name}" # 过滤出指定名称的容器
    ]
    try:
        result = _run_docker_command(command)
        if result.stdout.strip():
            print(f"Container '{container_name}' is running (ID: {result.stdout.strip()}).")
            return True
        else:
            return False
    except Exception:
        raise

# def is_docker_service_ready(
#     container_name: str,
#     timeout: int = 300,
#     polling_interval: int = 5
# ) -> bool:
#     """
#     检查指定名称的 Docker 容器内的服务是否已启动并准备好工作，
#     通过尝试连接到服务URL中的主机和端口。

#     :param container_name: Docker 容器的名称。
#     :param timeout: 等待服务准备就绪的总超时时间（秒）。
#     :param polling_interval: 检查服务状态的间隔时间（秒）。
#     :return: 如果容器在超时时间内运行且端口开放，则返回 True；否则返回 False。
#     """
#     import socket
#     from urllib.parse import urlparse
#     parsed_url = urlparse("http://127.0.0.1:8383")
#     host = parsed_url.hostname
#     port = parsed_url.port

#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         # 1. 检查容器是否正在运行
#         if not is_docker_container_running(container_name):
#             print(f"Container '{container_name}' is not running. Waiting {polling_interval}s...")
#             time.sleep(polling_interval)
#             continue

#         # 2. 容器正在运行，尝试检查端口是否开放
#         print(f"Attempting to connect to {host}:{port}...")
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#                 s.settimeout(polling_interval) # 设置连接超时，避免长时间阻塞
#                 # connect_ex() 返回 0 表示成功，否则是错误码
#                 result = s.connect_ex((host, port))
#                 if result == 0:
#                     print(f"Port {port} on {host} is open for '{container_name}'. Service is deemed ready.")
#                     return True
#                 else:
#                     # errno.ECONNREFUSED (111 on Linux) if nothing listening
#                     print(f"Port {port} on {host} is not open for '{container_name}' (Error code: {result}). Waiting {polling_interval}s...")
#         except Exception as e:
#             # 捕获其他任何socket相关的错误（如主机名解析失败等）
#             print(f"An unexpected error occurred during port check for {host}:{port}: {e}. Waiting {polling_interval}s...")

#         time.sleep(polling_interval)

#     print(f"Timeout ({timeout}s) reached. Service in '{container_name}' did not become ready (port not open).")
#     return False

def docker_container_exists(container_name):
    """
    检查指定名称的 Docker 容器是否存在（无论运行或停止）。
    """
    print(f"Checking if container '{container_name}' exists...")
    command = [
        "docker", "ps",
        "-aq", # 输出所有容器ID，包括停止的
        "--filter", f"name={container_name}" # 过滤出指定名称的容器
    ]
    try:
        result = _run_docker_command(command)
        if result.stdout.strip():
            print(f"Container '{container_name}' exists (ID: {result.stdout.strip()}).")
            return True
        else:
            print(f"Container '{container_name}' does not exist.")
            return False
    except Exception:
        return False # 如果执行命令本身出错，就认为不存在

def start_heygem_service(volume_host_path):
    """
    启动 heygem-gen-video Docker 容器/服务。
    优先尝试直接启动已存在的容器，否则通过 docker-compose 创建和启动。
    """
    print(f"\n--- Attempting to start '{CONTAINER_NAME}' service ---")
    
    # 优先级 1: 检查同名容器是否已经运行
    if is_docker_container_running(CONTAINER_NAME):
        print(f"\nContainer '{CONTAINER_NAME}' is already running. No action needed.")
        return True

    # 优先级 2: 检查同名容器是否存在但已停止，并尝试直接启动
    if docker_container_exists(CONTAINER_NAME):
        print(f"\nContainer '{CONTAINER_NAME}' exists but is stopped. Attempting to start it directly...")
        start_command = ["docker", "start", CONTAINER_NAME]
        try:
            _run_docker_command(start_command, check_result=True)
            print(f"Successfully sent 'docker start {CONTAINER_NAME}' command.")
            
            time.sleep(5)
            # 验证是否启动成功
            if is_docker_container_running(CONTAINER_NAME):
                print(f"Container '{CONTAINER_NAME}' confirmed as running after direct start.")
                return True
            else:
                print(f"Container '{CONTAINER_NAME}' did not become ready after direct start.")
                return False

        except Exception:
            print(f"Failed to directly start container '{CONTAINER_NAME}'. Falling back to docker-compose.", file=sys.stderr)
            # 如果直接启动失败，继续尝试使用 docker-compose (可能容器损坏或其他问题)
            pass 

    # 优先级 3: 如果容器不存在或直接启动失败，使用 docker-compose 创建并启动
    print(f"\nContainer '{CONTAINER_NAME}' not found or failed direct start. Using docker-compose to create and start it...")
    print(f"Docker Compose file: {DOCKER_COMPOSE_FILE}")
    print(f"Volume host path to use: {volume_host_path}")

    env_for_subprocess = os.environ.copy()
    env_for_subprocess[VOLUME_ENV_VAR] = volume_host_path

    compose_up_command = [
        "docker-compose",
        "-f",
        DOCKER_COMPOSE_FILE,
        "up",
        "-d"
    ]

    try:
        print(f"\nExecuting command: {' '.join(compose_up_command)}")
        result = _run_docker_command(compose_up_command, check_result=True, env=env_for_subprocess)
        print("\nDocker Compose Output (stdout):")
        print(result.stdout)
        if result.stderr:
            print("\nDocker Compose Output (stderr):")
            print(result.stderr)
        
        print(f"\nSuccessfully executed 'docker-compose up -d' for '{CONTAINER_NAME}'.")

        # 验证是否启动成功
        time_sec = 0
        while time_sec < 7200:
            if is_docker_container_running(CONTAINER_NAME):
                print(f"Container '{CONTAINER_NAME}' confirmed as running after direct start.")
                return True

            print(f"Waiting for container '{CONTAINER_NAME}' to start... Downloading image and creating... ")
            time.sleep(20) # 每10秒检查一次
            time_sec += 20

    except Exception:
        print(f"\nFailed to start '{CONTAINER_NAME}' using docker-compose. Please check the logs above for details.", file=sys.stderr)
        return False

def _get_container_id(container_name):
    """
    获取指定名称的容器ID，如果容器不存在或不运行，则返回None。
    """
    command = [
        "docker", "ps",
        "-q",  # 只输出容器ID
        "--filter", f"name={container_name}" # 过滤出指定名称的容器
    ]
    try:
        result = _run_docker_command(command)
        container_id = result.stdout.strip()
        if container_id:
            return container_id
        return None
    except Exception as e:
        print(f"Error getting container ID for '{container_name}': {e}", file=sys.stderr)
        return None
       
def stop_heygem_service():
    """
    停止 heygem-gen-video Docker 容器。
    优先尝试使用 'docker stop <container_name>' 命令。
    如果服务未运行，则跳过停止操作。
    """
    print(f"\n--- Attempting to stop '{CONTAINER_NAME}' service ---")

    # 检查容器是否运行 (无论是 compose 启动的还是直接启动的)
    current_container_id = _get_container_id(CONTAINER_NAME)
    if not current_container_id:
        print(f"\nContainer '{CONTAINER_NAME}' is not running. No need to stop.")
        return True

    print(f"\nContainer '{CONTAINER_NAME}' is running (ID: {current_container_id}). Attempting to stop it...")

    # 尝试使用 docker stop 命令
    stop_command = ["docker", "stop", current_container_id] # 使用获取到的ID更精确

    try:
        print(f"\nExecuting command: {' '.join(stop_command)}")
        _run_docker_command(stop_command, check_result=True)
        print(f"\nSuccessfully sent 'docker stop {CONTAINER_NAME}' command.")

    except Exception as e:
        print(f"\nFailed to stop '{CONTAINER_NAME}' using 'docker stop': {e}. Please check the logs above.", file=sys.stderr)

def save_tensor_as_video_lossless(image_tensor, output_path, fps=24):
    """
    将形状为 [frames, height, width, channels] 的 PyTorch 张量保存为视频文件，
    尝试最小化损失，特别是使用无损编码器。

    :param image_tensor: 形状为 [frames, height, width, channels] 的 PyTorch 张量，范围为 [0, 1]
    :param output_path: 输出视频文件路径 (建议使用 .mkv, .avi 或 .mov 扩展名以支持更多无损编码)
    :param fps: 视频帧率
    """
    # 确保张量在 CPU 上
    if image_tensor.is_cuda:
        tensor_np = image_tensor.cpu().numpy()
    else:
        tensor_np = image_tensor.numpy()

    # 将张量值从 [0, 1] 范围转换为 [0, 255] 范围，并确保数据类型为 uint8
    # 使用 np.round 进行四舍五入，并使用 np.clip 确保在 [0, 255] 范围内
    tensor_np = np.clip(np.round(tensor_np * 255), 0, 255).astype(np.uint8)

    # 尝试使用FFV1 (无损)
    try:
        with imageio.get_writer(output_path, fps=fps, codec='ffv1', macro_block_size=1) as writer:
            # macro_block_size=1 可以提高编码器精确度，可能略微增大文件，但FFV1已经是无损了
            for frame in tensor_np:
                writer.append_data(frame)
        print(f"Successfully saved video to {output_path} using FFV1 (lossless).")
    except Exception as e:
        print(f"Failed to use FFV1 codec: {e}. Trying libx264 with crf=0 (lossless, but might have YUV conversion).")
        # 尝试使用 libx264, crf=0 (无损)
        try:
            # 'crf=0' 参数通过 ffmpeg_params 传递
            # 对于某些imageio版本，可能需要显式指定像素格式
            with imageio.get_writer(output_path, fps=fps, codec='libx264', ffmpeg_params=['-crf', '0', '-pix_fmt', 'yuv444p']) as writer:
                for frame in tensor_np:
                    writer.append_data(frame)
            print(f"Successfully saved video to {output_path} using libx264 (crf=0, lossless).")
        except Exception as e_h264:
            print(f"Failed to use libx264 crf=0: {e_h264}. Falling back to default (possibly lossy).")
            # 最后的退路：使用默认设置，可能会有损
            with imageio.get_writer(output_path, fps=fps) as writer:
                for frame in tensor_np:
                    writer.append_data(frame)
            print(f"Saved video to {output_path} using default imageio settings (potentially lossy).")

# def save_tensor_as_image_sequence(image_tensor, output_dir, filename_prefix="frame", fps=24):
#     """
#     将形状为 [frames, height, width, channels] 的 PyTorch 张量保存为无损图像序列。

#     :param image_tensor: 形状为 [frames, height, width, channels] 的 PyTorch 张量，范围为 [0, 1]
#     :param output_dir: 输出图像序列的目录
#     :param filename_prefix: 文件名前缀
#     :param fps: 视频帧率 (仅用于信息，不影响保存)
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     if image_tensor.is_cuda:
#         tensor_np = image_tensor.cpu().numpy()
#     else:
#         tensor_np = image_tensor.numpy()

#     # 量化到 uint8，四舍五入并裁剪
#     tensor_np = np.clip(np.round(tensor_np * 255), 0, 255).astype(np.uint8)

#     for i, frame in enumerate(tensor_np):
#         output_path = os.path.join(output_dir, f"{filename_prefix}_{i:05d}.png")
#         imageio.imwrite(output_path, frame)
#     print(f"Saved frames as PNG sequence to '{output_dir}/{filename_prefix}_XXXXX.png'.")
#     print(f"You can combine them into a video using FFmpeg: ffmpeg -i {output_dir}/{filename_prefix}_%05d.png -vf fps={fps} output.mkv")

# def load_image_sequence_to_tensor(input_dir, filename_pattern="frame_%05d.png"):
#     """
#     从无损图像序列加载图像并转换为形状为 [frames, height, width, channels] 的 PyTorch 张量。

#     :param input_dir: 包含图像序列的目录
#     :param filename_pattern: 图像文件名模式，例如 "frame_%05d.png"
#     :return: 形状为 [frames, height, width, channels] 的 PyTorch 张量
#     """
#     import glob
#     import re

#     files = sorted(glob.glob(os.path.join(input_dir, filename_pattern.replace('%05d', '*'))))
    
#     frames = []
#     for file_path in files:
#         # 确保按数字顺序读取
#         match = re.search(r'(\d+)\.png$', os.path.basename(file_path))
#         if match:
#             idx = int(match.group(1))
#             # 读取并归一化
#             frame = imageio.imread(file_path)
#             frame_normalized = (frame.astype(np.float32) / 255.0)
#             frames.append((idx, frame_normalized))
    
#     # 确保按索引排序
#     frames.sort(key=lambda x: x[0])
#     frames_np = np.stack([f[1] for f in frames], axis=0)
#     tensor = torch.tensor(frames_np, dtype=torch.float32)
#     print(f"Loaded {len(frames)} frames from '{input_dir}' as tensor.")
#     return tensor

def video_to_tensor(video_path):
    """
    将本地视频文件转换为形状为 [frames, height, width, channels] 的 PyTorch 张量，
    范围为 [0, 1]。

    :param video_path: 本地视频文件路径
    :return: 形状为 [frames, height, width, channels] 的 PyTorch 张量
    """
    # 使用 imageio 读取视频文件
    reader = imageio.get_reader(video_path)

    frames = []
    for frame in reader:
        # 将每一帧从 uint8 类型转换为 float32 类型，并归一化到 [0, 1] 范围
        frame_normalized = (frame.astype(np.float32) / 255.0)
        frames.append(frame_normalized)

    # 将列表转换为 NumPy 数组
    frames_np = np.stack(frames, axis=0)

    # 将 NumPy 数组转换为 PyTorch 张量
    tensor = torch.tensor(frames_np, dtype=torch.float32)

    return tensor