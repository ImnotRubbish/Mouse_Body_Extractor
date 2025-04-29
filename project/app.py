import os
import yaml
from flask import Flask, render_template, request, redirect, url_for
from dlc_module import DLCProcessor

app = Flask(__name__)
processor = DLCProcessor()

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/create_project', methods=['GET', 'POST'])
def create_project():
    if request.method == 'POST':
        project_name = request.form['project_name']
        user_name = request.form['user_name']
        video_dir = request.form['video_dir']
        work_dir = request.form['work_dir']

        config_path, output_messages = processor.create_project(video_dir, work_dir, project_name, user_name)
        processor.config_path = config_path
        if config_path:
            message = f"项目创建成功！配置文件地址: {config_path}\n" + "\n控制台输出:\n"+ "\n".join(output_messages)
        else:
            message = "项目创建失败，请检查输入信息。\n" + "\n控制台输出:\n" + "\n".join(output_messages)
        return render_template('create_project.html', message=message)
    return render_template('create_project.html')

@app.route('/load_project', methods=['GET', 'POST'])
def load_project():
    if request.method == 'POST':
        config_path = request.form['config_path']
        try:
            processor.load_project(config_path)
            message = f"项目加载成功！配置文件路径: {config_path}"
        except Exception as e:
            message = f"项目加载失败: {str(e)}"
        return render_template('load_project.html', message=message)
    return render_template('load_project.html')

@app.route('/work')
def work():
    return render_template('work.html')

@app.route('/modify_config', methods=['GET', 'POST'])
def modify_config():
    if processor.config_path:
        with open(processor.config_path, 'r') as file:
            config = yaml.safe_load(file)
        bodyparts = config.get('bodyparts', [])
        skeleton = config.get('skeleton', [])
        dotsize = config.get('dotsize', 5)
        skeleton_color = config.get('skeleton_color', 'red')

        if request.method == 'POST':
            new_bodyparts = [part for part in request.form.getlist('bodyparts') if part.strip()]
            new_skeleton = []
            skeleton_part1 = request.form.getlist('skeleton_part1')
            skeleton_part2 = request.form.getlist('skeleton_part2')
            for i in range(0, len(skeleton_part1)):
                if skeleton_part1[i].strip() and skeleton_part2[i].strip():
                    new_skeleton.append([skeleton_part1[i], skeleton_part2[i]])
            new_dotsize = int(request.form.get('dotsize', 5))
            new_skeleton_color = request.form.get('skeleton_color', 'red')

            try:
                processor.update_config(
                    bodyparts=new_bodyparts,
                    skeleton=new_skeleton,
                    dotsize=new_dotsize,
                    skeleton_color=new_skeleton_color
                )
                message = "配置文件保存成功！"
            except Exception as e:
                message = f"配置文件保存失败: {str(e)}"
            return render_template('modify_config.html', bodyparts=bodyparts, skeleton=skeleton, dotsize=dotsize, skeleton_color=skeleton_color, message=message)
        return render_template('modify_config.html', bodyparts=bodyparts, skeleton=skeleton, dotsize=dotsize, skeleton_color=skeleton_color)
    return render_template('modify_config.html', message="请先创建或加载项目")

@app.route('/extract_frames', methods=['GET', 'POST'])
def extract_frames():
    if request.method == 'POST':
        min_interval = int(request.form.get('min_interval', 30))
        max_interval = int(request.form.get('max_interval', 40))
        try:
            output_messages = processor.extract_frames(min_interval, max_interval)
            message = "视频帧提取成功！\n控制台输出:\n" + "\n".join(output_messages)
        except Exception as e:
            message = f"视频帧提取失败: {str(e)}"
        return render_template('extract_frames.html', message=message)
    return render_template('extract_frames.html')

@app.route('/label_frames', methods=['GET', 'POST'])
def label_frames():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'start':
            try:
                output_messages = processor.label_frames()
                message = "控制台输出:\n" + "\n".join(output_messages)
            except Exception as e:
                message = f"标记点注册失败: {str(e)}"
        elif action == 'check':
            try:
                output_messages = processor.check_labels()
                message = "标记点注册检查成功！\n控制台输出:\n" + "\n".join(output_messages)
            except Exception as e:
                message = f"标记点注册检查失败: {str(e)}"
        return render_template('label_frames.html', message=message)
    return render_template('label_frames.html')

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'create_dataset':
            try:
                output_messages = processor.create_training_dataset()
                message = "数据集创建成功！\n控制台输出:\n" + "\n".join(output_messages)
            except Exception as e:
                message = f"数据集创建失败: {str(e)}"
        elif action == 'start_train':
            try:
                gpu_output = processor._setup_gpu()
                output_messages = processor.train_model()
                message = "模型训练成功！\nGPU设置输出:\n" + "\n".join(gpu_output) + "\n控制台输出:\n" + "\n".join(output_messages)
            except Exception as e:
                message = f"模型训练失败: {str(e)}"
        elif action == 'evaluate':
            try:
                output_messages = processor.evaluate_model()
                message = "模型评估成功！\n控制台输出:\n" + "\n".join(output_messages)
            except Exception as e:
                message = f"模型评估失败: {str(e)}"
        return render_template('train_model.html', message=message)
    return render_template('train_model.html')

@app.route('/export_video', methods=['GET', 'POST'])
def export_video():
    if processor.config_path:
        video_dir = processor.config_path.split('config.yaml')[0] + 'videos'
        save_type = '.mp4'

        if request.method == 'POST':
            video_dir = request.form.get('video_dir', video_dir)
            save_type = request.form.get('save_type', save_type)
            action = request.form.get('action')
            if action == 'process':
                try:
                    output_messages = processor.process_videos(video_dir)
                    message = "视频处理成功！\n控制台输出:\n" + "\n".join(output_messages)
                except Exception as e:
                    message = f"视频处理失败: {str(e)}"
            elif action == 'export':
                try:
                    output_messages = processor.create_labeled_video(video_dir, save_type)
                    message = "标记视频导出成功！\n控制台输出:\n" + "\n".join(output_messages)
                except Exception as e:
                    message = f"标记视频导出失败: {str(e)}"
            return render_template('export_video.html', video_dir=video_dir, save_type=save_type, message=message)
        return render_template('export_video.html', video_dir=video_dir, save_type=save_type)
    return render_template('export_video.html', message="请先创建或加载项目")

if __name__ == '__main__':
    
    app.run(debug=False)