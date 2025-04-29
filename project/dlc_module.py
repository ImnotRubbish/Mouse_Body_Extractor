import cv2
import os
import io
import sys
import shutil
import random
import time
import yaml
import tensorflow as tf
import deeplabcut as dlc

class DLCProcessor:
    def __init__(self, config_path=None):
        self.config_path = config_path

    def load_project(self, config_path):
        """加载已存在的项目配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        if not config_path.endswith('config.yaml'):
            raise ValueError("配置文件必须以 'config.yaml' 结尾")
        
        self.config_path = config_path
        print(f"项目已成功加载: {config_path}")

    def create_project(self, video_dir, work_dir, project_name, user_name):
        all_video = []
        output_messages = []  # 用于存储输出信息
    
        def collect_videos(path):
            try:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        collect_videos(item_path)
                    elif item.lower().endswith(('.mp4', '.avi', '.mov')):
                        all_video.append(item_path)
            except FileNotFoundError:
                error_msg = f"文件夹 {path} 未找到。"
                print(error_msg)
                output_messages.append(error_msg)
    
        collect_videos(video_dir)
    
        try:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            self.config_path = dlc.create_new_project(project_name, user_name, all_video, 
                                                    working_directory=work_dir, copy_videos=True, 
                                                    multianimal=False)
            
            # 获取输出内容
            output = sys.stdout.getvalue()
            output_messages.extend(output.splitlines())
            
            # 恢复标准输出
            sys.stdout = old_stdout
            
            # 返回配置路径和输出信息
            return self.config_path, output_messages
        except Exception as e:
            error_msg = f"创建项目时出现错误: {e}"
            print(error_msg)
            output_messages.append(error_msg)
            return None, output_messages

    def extract_frames(self, min_interval=3500, max_interval=3600):
        output_messages = []
        if self.config_path:
            video_dir = self.config_path.split('config.yaml')[0] + 'videos'
            save_dir = self.config_path.split('config.yaml')[0] + 'labeled-data'
        else:
            output_messages.append("请先创建项目")
            return output_messages
        
        for video in os.listdir(video_dir):
            video_name = os.path.join(video_dir, video)
            video_ext = os.path.splitext(video)[1].lower()
            if video_ext not in ('.mp4', '.avi', '.mov'):
                continue
                
            interval = random.randint(min_interval, max_interval)
            save_path = os.path.join(save_dir, video.split(video_ext)[0]) + '\\'
            output_messages.append(f"保存路径: {save_path}")
            self._get_frame_from_video(video_name, interval, save_path)
        
        return output_messages

    def _get_frame_from_video(self, video_name, interval, path='./'):
        output_messages = []
        save_path = video_name.split('.mp4')[0] + '/'
        is_exists = os.path.exists(path)
        if not is_exists:
            os.makedirs(path)
            output_messages.append(f"路径 {path} 已创建")

        video_capture = cv2.VideoCapture(video_name)
        i = 0

        while True:
            success, frame = video_capture.read()
            i += 1
            if i % interval == 0:
                digits=len(str(i))
                save_name = path + 'img' + (5-digits)*'0' + str(i) + '.png'
                cv2.imwrite(save_name, frame)
                output_messages.append(f"图片 {save_name} 已保存")
            if not success:
                output_messages.append("视频已全部读取")
                break
        
        return output_messages

    def label_frames(self):
        output_messages = []
        if self.config_path:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            dlc.label_frames(self.config_path)
            
            output = sys.stdout.getvalue()
            output_messages.extend(output.splitlines())
            sys.stdout = old_stdout
        else:
            output_messages.append("请先创建项目")
        
        return output_messages

    def check_labels(self):
        output_messages = []
        if self.config_path:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            dlc.check_labels(self.config_path, draw_skeleton=True, visualizeindividuals=True)
            
            output = sys.stdout.getvalue()
            output_messages.extend(output.splitlines())
            sys.stdout = old_stdout
        else:
            output_messages.append("请先创建项目")
        
        return output_messages

    def create_training_dataset(self):
        output_messages = []
        if self.config_path:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            dlc.create_training_dataset(self.config_path)
            
            output = sys.stdout.getvalue()
            output_messages.extend(output.splitlines())
            sys.stdout = old_stdout
        else:
            output_messages.append("请先创建项目")
        
        return output_messages

    def train_model(self, maxiters=50000, displayiters=5000):
        output_messages = []
        if self.config_path:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            start_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            output_messages.append(f"训练开始时间: {start_time}")
            
            dlc.train_network(self.config_path, maxiters=maxiters, displayiters=displayiters, allow_growth=True)
            
            end_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            output_messages.append(f"训练结束时间: {end_time}")
            
            output = sys.stdout.getvalue()
            output_messages.extend(output.splitlines())
            sys.stdout = old_stdout
        else:
            output_messages.append("请先创建项目")
        
        return output_messages

    def _setup_gpu(self):
        output_messages = []
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                output_messages.append(f"{len(gpus)} 物理GPU, {len(logical_gpus)} 逻辑GPU")
            except RuntimeError as e:
                output_messages.append(str(e))
        
        return output_messages

    def evaluate_model(self):
        output_messages = []
        if self.config_path:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            dlc.evaluate_network(self.config_path, Shuffles=[1], plotting=True)
            
            output = sys.stdout.getvalue()
            output_messages.extend(output.splitlines())
            sys.stdout = old_stdout
        else:
            output_messages.append("请先创建项目")
        
        return output_messages

    def process_videos(self, video_dir=None):
        output_messages = []
        if self.config_path:
            if video_dir is None:
                video_dir = self.config_path.split('config.yaml')[0] + 'videos'
            all_video = [os.path.join(video_dir, video) for video in os.listdir(video_dir)]
            
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            dlc.analyze_videos(self.config_path, all_video, videotype='.mp4', save_as_csv=True)
            
            output = sys.stdout.getvalue()
            output_messages.extend(output.splitlines())
            sys.stdout = old_stdout
        else:
            output_messages.append("请先创建项目")
        
        return output_messages

    def create_labeled_video(self, video_dir=None, save_type='.mp4'):
        output_messages = []
        if self.config_path:
            if video_dir is None:
                video_dir = self.config_path.split('config.yaml')[0] + 'videos'
            all_video = [os.path.join(video_dir, video) for video in os.listdir(video_dir)]
            
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            dlc.create_labeled_video(self.config_path, all_video, videotype=save_type, draw_skeleton=True)
            
            output = sys.stdout.getvalue()
            output_messages.extend(output.splitlines())
            sys.stdout = old_stdout
        else:
            output_messages.append("请先创建项目")
        
        return output_messages

    def update_config(self, bodyparts=None, skeleton=None, dotsize=None, skeleton_color=None):
        output_messages = []
        if not self.config_path:
            output_messages.append("请先创建项目")
            return output_messages
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            if bodyparts is not None:
                config['bodyparts'] = bodyparts
            if skeleton is not None:
                config['skeleton'] = skeleton
            if dotsize is not None:
                config['dotsize'] = dotsize
            if skeleton_color is not None:
                config['skeleton_color'] = skeleton_color
            
            with open(self.config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            
            output_messages.append("配置文件更新成功")
        except Exception as e:
            output_messages.append(f"更新配置文件时出错: {e}")
        
        return output_messages

if __name__ == "__main__":
    # 初始化
    processor = DLCProcessor()

    # 创建项目
    config_path = processor.create_project(
        video_dir=r'D:\project\Mouse_Body_Extractor\try\video',
        work_dir=r'D:\project\Mouse_Body_Extractor\try',
        project_name='ketamine222',
        user_name='RXH'
    )

    # processor.update_config(
    #     bodyparts=['nose','leftear','rightear','head','leftarm','rightarm','leftleg','rightleg','centerofmass','tailbase','tailcenter','tailend'],  # 替换为你的新身体部分列表
    #     skeleton=[['head', 'nose'], ['head', 'leftear'], ['head', 'rightear'], ['head', 'centerofmass'], ['centerofmass', 'leftarm'], ['centerofmass', 'rightarm'], ['centerofmass', 'leftleg'], ['centerofmass', 'rightleg'], ['centerofmass', 'tailbase'], ['tailbase', 'tailcenter'], ['tailcenter', 'tailend']],  # 替换为你的新骨架列表
    #     dotsize=10,  # 替换为你的新点大小
    #     skeleton_color='black'  # 替换为你的新骨架颜色列表,
    # )

    # # 提取帧
    # processor.extract_frames(
    #     min_interval=30,
    #     max_interval=40
    # )

    # # 标记帧
    # processor.label_frames()

    # # 检查标签
    # processor.check_labels()

    # # 创建训练数据集
    # processor.create_training_dataset()

    # # 设置 GPU
    # processor._setup_gpu()

    # # 训练模型
    # processor.train_model()

    # # 评估模型
    # processor.evaluate_model()

    # # 处理视频
    # processor.process_videos(save_type='.mp4')