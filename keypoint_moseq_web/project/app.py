from flask import Flask, render_template, redirect, url_for
import os
import keypoint_moseq as kpms

app = Flask(__name__)

# 用于存储中间结果的全局变量（简化版）
state = {
    "project_dir": None,
    "keypoint_data_path": None,
    "coordinates": None,
    "confidences": None,
    "bodyparts": None,
    "config": None,
    "data": None,
    "metadata": None,
    "pca": None,
    "model": None,
    "model_name": None,
    "results": None
}

# 示例路径和参数（可以根据需要调整）
PROJECT_DIR = "my_project"
VIDEO_DIR = "videos"
KEYPOINT_DATA_PATH = "keypoints.h5"
BODYPARTS = ["nose", "left_ear", "right_ear", "neck", "spine1", "spine2", "spine3", "tailbase"]
SKELETON = [["nose", "neck"], ["neck", "spine1"], ["spine1", "spine2"], ["spine2", "spine3"], ["spine3", "tailbase"],
            ["neck", "left_ear"], ["neck", "right_ear"]]
CONFIG = {
    "low_dim_size": 4,
    "fps": 30,
    "sleap_version": "1.2.7",
    "deeplabcut_version": "2.2.0"
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/create_project')
def create_project():
    state["project_dir"] = PROJECT_DIR
    kpms.setup_project(PROJECT_DIR, video_dir=VIDEO_DIR, bodyparts=BODYPARTS, skeleton=SKELETON)
    return render_template('setup_completed.html')


@app.route('/load_keypoints')
def load_keypoints():
    result = kpms.load_keypoints(KEYPOINT_DATA_PATH, 'deeplabcut')
    state.update({
        "coordinates": result['coordinates'],
        "confidences": result['confidences'],
        "bodyparts": result['bodyparts']
    })
    return render_template('keypoints_loaded.html')


@app.route('/format_data')
def format_data():
    result = kpms.format_data(state["coordinates"], state["confidences"], low_dim_size=4)
    state.update({
        "data": result[0],
        "metadata": result[1]
    })
    return render_template('data_formatted.html')


@app.route('/run_pca_analysis')
def run_pca_analysis():
    project_dir = state["project_dir"]
    pca = kpms.fit_pca(**state["data"], low_dim_size=4)
    kpms.save_pca(pca, project_dir)

    kpms.plot_scree(pca, project_dir=project_dir)
    kpms.plot_pcs(pca, project_dir=project_dir, low_dim_size=4)

    state["pca"] = pca
    return render_template('pca_analysis_complete.html')


@app.route('/update_config')
def update_config():
    kpms.update_config(state["project_dir"], latent_dim=4)
    return redirect(url_for('initialize_model_view'))


@app.route('/initialize_model')
def initialize_model_view():
    model = kpms.init_model(state["data"], pca=state["pca"], low_dim_size=4)
    state["model"] = model
    return redirect(url_for('fit_model_view'))


@app.route('/fit_model')
def fit_model_view():
    model, model_name = kpms.fit_model(
        state["model"], state["data"], state["metadata"], state["project_dir"],
        ar_only=True, num_iters=50, start_iter=0
    )
    state["model"] = model
    state["model_name"] = model_name
    return render_template('model_fitted.html')


@app.route('/extract_results')
def extract_results_view():
    results = kpms.extract_results(state["model"], state["metadata"], state["project_dir"], state["model_name"])
    kpms.save_results_as_csv(results, state["project_dir"], state["model_name"])
    state["results"] = results
    return render_template('results_extracted.html')


@app.route('/trajectory_plots')
def trajectory_plots_view():
    kpms.generate_trajectory_plots(
        state["coordinates"], state["results"], state["project_dir"], state["model_name"], low_dim_size=4
    )
    return render_template('trajectory_plots.html')


@app.route('/grid_movies')
def grid_movies_view():
    kpms.generate_grid_movies(state["results"], state["project_dir"], state["model_name"], coordinates=state["coordinates"], low_dim_size=4)
    return redirect(url_for('home'))


@app.route('/similarity_dendrogram')
def similarity_dendrogram_view():
    kpms.plot_similarity_dendrogram(
        state["coordinates"], state["results"], state["project_dir"], state["model_name"], low_dim_size=4
    )
    return redirect(url_for('home'))