import os
import shutil
import html
import copy
import re
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# Offline WebUI keeps the CUDA Graph implementation available, but remains
# eager by default. Set RVC_OFFLINE_CUDA_GRAPH=1 to opt in for benchmarking or
# controlled deployments.
_offline_cuda_graph = os.environ.get("RVC_OFFLINE_CUDA_GRAPH", "0") == "1"
os.environ["RVC_CUDA_GRAPH"] = "1" if _offline_cuda_graph else "0"

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("no_proxy", "localhost, 127.0.0.1, ::1")
os.environ.setdefault("weight_root", "assets/weights")
os.environ.setdefault("weight_pymss_root", "assets/pymss_weights")
os.environ.setdefault("index_root", "logs")
os.environ.setdefault("outside_index_root", "assets/indices")
os.environ.setdefault("rmvpe_root", "assets/rmvpe")

now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
for name in os.listdir(tmp):
    if name == "jieba.cache":
        continue
    path = os.path.join(tmp, name)
    delete = (
        os.remove if os.path.isfile(path) or os.path.islink(path) else shutil.rmtree
    )
    try:
        delete(path)
    except Exception as error:
        print(str(error))

from configs.config import Config
from infer.vc.modules import VC
from infer.vc.utils import get_index_path_from_model
from tools.pymss_webui import (
    PYMSS_MODEL_CHOICES,
    get_model_info,
    pymss_separate as _pymss_separate_core,
    stop_pymss_separation as _stop_pymss_separation_core,
)
from tools.download_models import (
    BASE_ASSETS,
    download_base_assets,
    download_voice_model_url,
)
from tools.file_io import read_text
from i18n.i18n import I18nAuto
import torch
import gradio as gr
import traceback
import logging
import socket


logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def find_available_port(start_port, host="0.0.0.0"):
    """Return the first bindable TCP port at or above ``start_port``."""
    if not 1 <= start_port <= 65535:
        raise ValueError(f"Port must be between 1 and 65535, got {start_port}.")

    for port in range(start_port, 65536):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
            return port
        except OSError:
            continue

    raise OSError(
        f"No available TCP port from {start_port} through 65535; WebUI was not started."
    )


def is_gradio_port_in_use_error(error, port):
    """Recognize Gradio's explicit-port conflict without hiding other launch errors."""
    return str(error).startswith(f"Port {port} is in use.")


def launch_webui_with_port_fallback(app, config):
    """Launch Gradio, increasing the requested port until startup succeeds."""
    next_port = config.listen_port
    queued_app = app.queue(concurrency_count=511, max_size=1022)
    while True:
        config.listen_port = find_available_port(next_port)
        if config.listen_port != next_port:
            logger.warning(
                "Port %s is occupied; trying port %s instead.",
                next_port,
                config.listen_port,
            )
        try:
            queued_app.launch(
                server_name="0.0.0.0",
                inbrowser=not config.noautoopen,
                server_port=config.listen_port,
                quiet=True,
            )
            return config.listen_port
        except OSError as error:
            if not is_gradio_port_in_use_error(error, config.listen_port):
                raise
            if config.listen_port == 65535:
                raise OSError(
                    "No available TCP port through 65535; WebUI was not started."
                ) from error
            logger.warning(
                "Port %s became occupied while Gradio was starting; trying the next port.",
                config.listen_port,
            )
            next_port = config.listen_port + 1

runtime_dirs = (
    os.path.join(now_dir, "logs"),
    os.environ["weight_root"],
    os.environ["weight_pymss_root"],
    os.environ["index_root"],
    os.environ["outside_index_root"],
    os.environ["rmvpe_root"],
    os.path.join(now_dir, "assets", "hubert_base"),
    os.path.join(now_dir, "assets", "pretrained"),
    os.path.join(now_dir, "assets", "pretrained_v2"),
)
for runtime_dir in runtime_dirs:
    os.makedirs(runtime_dir, exist_ok=True)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
vc = VC(config)


i18n = I18nAuto()
logger.info(i18n)
print(
    i18n("当前设备：%s | 推理精度：%s") % (config.device, config.dtype),
    flush=True,
)
# Inference-only build: device selection is handled inside configs.config.
config = Config()
vc = VC(config)


i18n = I18nAuto()
logger.info(i18n)
print(
    i18n("当前设备：%s | 推理精度：%s") % (config.device, config.dtype),
    flush=True,
)


weight_root = os.getenv("weight_root")
weight_pymss_root = os.getenv("weight_pymss_root")
outside_index_root = os.getenv("outside_index_root")

def weight_names():
    return sorted(
        name for name in os.listdir(weight_root) if name.endswith(".pth")
    )


def refresh_weight_choices(previous_names=None, force=False):
    current_names = tuple(weight_names())
    if force or current_names != previous_names:
        return current_names, change_choices()
    return current_names, {"__type__": "update"}


names = weight_names()
pymss_names = PYMSS_MODEL_CHOICES


def change_choices():
    return {"choices": weight_names(), "__type__": "update"}


def clean():
    return {"value": "", "__type__": "update"}


def selected_speaker_id(slider_value, dropdown_value):
    value = dropdown_value if dropdown_value is not None else slider_value
    if isinstance(value, str):
        match = re.search(r"ID\s*[:：]\s*(\d+)\s*[)）]?\s*$", value)
        if match:
            return int(match.group(1))
    return int(value)


def normalize_index_path(file_index):
    return (
        str(file_index or "")
        .strip(" ")
        .strip('"')
        .strip("\n")
        .strip('"')
        .strip(" ")
        .replace("trained", "added")
    )


def report_missing_index(file_index):
    index_path = normalize_index_path(file_index)
    if index_path and not os.path.isfile(index_path):
        message = i18n("索引文件不存在，将不使用索引继续推理：%s") % index_path
        print(message, flush=True)
        raise gr.Error(message)


def update_speaker_index(model_name, slider_value, dropdown_value):
    try:
        speaker_id = selected_speaker_id(slider_value, dropdown_value)
    except (TypeError, ValueError):
        speaker_id = None
    path = get_index_path_from_model(model_name, speaker_id)
    update = {"value": path, "__type__": "update"}
    return update, dict(update)


def update_dropdown_speaker_index(model_name, dropdown_value):
    return update_speaker_index(model_name, 0, dropdown_value)


def vc_single_with_speaker(slider_value, dropdown_value, *args):
    return vc.vc_single(selected_speaker_id(slider_value, dropdown_value), *args)


def vc_multi_with_speaker(slider_value, dropdown_value, *args):
    yield from vc.vc_multi(selected_speaker_id(slider_value, dropdown_value), *args)


def button_update(value=None, variant=None, visible=None):
    update = {"__type__": "update"}
    if value is not None:
        update["value"] = value
    if variant is not None:
        update["variant"] = variant
    if visible is not None:
        update["visible"] = visible
    return update


def render_pymss_progress(percent=0, label="等待开始", state="idle"):
    percent = max(0.0, min(100.0, float(percent or 0)))
    colors = {
        "idle": "#64748b",
        "running": "#2563eb",
        "done": "#15803d",
        "stopped": "#b45309",
        "failed": "#b91c1c",
    }
    color = colors.get(state, colors["running"])
    safe_label = html.escape(str(label or ""))
    return (
        '<div style="min-height:52px;padding:6px 0;">'
        '<div style="display:flex;justify-content:space-between;gap:12px;'
        'align-items:center;margin-bottom:7px;font-size:14px;line-height:20px;">'
        '<span style="overflow-wrap:anywhere;">%s</span>'
        '<strong style="flex:0 0 auto;color:%s;">%.1f%%</strong>'
        "</div>"
        '<div role="progressbar" aria-valuemin="0" aria-valuemax="100" '
        'aria-valuenow="%.1f" style="height:10px;width:100%%;overflow:hidden;'
        'border-radius:4px;background:#e2e8f0;">'
        '<div style="height:100%%;width:%.3f%%;background:%s;"></div>'
        "</div></div>"
    ) % (safe_label, color, percent, percent, percent, color)


def run_pymss_separation(
    model_name,
    inp_root,
    save_root_vocal,
    paths,
    save_root_ins,
    format0,
):
    progress_state = {
        "percent": 0.0,
        "label": "正在准备 PyMSS 分离任务",
        "state": "running",
    }
    busy = False

    def update_progress(event):
        nonlocal busy
        event_type = event.get("event")
        file_count = max(1, int(event.get("file_count") or 1))
        file_index = max(1, int(event.get("file_index") or 1))
        message = str(event.get("message") or "")

        if event_type == "progress":
            done = max(0.0, float(event.get("done") or 0))
            total = max(1.0, float(event.get("total") or 1))
            file_fraction = min(1.0, done / total)
            progress_state["percent"] = (
                (file_index - 1 + file_fraction) / file_count * 100
            )
            progress_state["label"] = "文件 %s/%s · %s · %.0f/%.0f 秒" % (
                file_index,
                file_count,
                message or "正在处理音频",
                done,
                total,
            )
            progress_state["state"] = "running"
        elif event_type == "file_start":
            progress_state["percent"] = (file_index - 1) / file_count * 100
            progress_state["label"] = message
            progress_state["state"] = "running"
        elif event_type == "file":
            progress_state["percent"] = file_index / file_count * 100
            progress_state["label"] = (
                message.splitlines()[0] if message else "文件处理结束"
            )
            progress_state["state"] = "running" if event.get("ok") else "failed"
        elif event_type == "done":
            successful = int(event.get("successful") or 0)
            failed = int(event.get("failed") or 0)
            progress_state["percent"] = 100.0
            progress_state["label"] = "分离完成：成功 %s，失败 %s" % (
                successful,
                failed,
            )
            progress_state["state"] = "done" if failed == 0 else "failed"
        elif event_type == "retry_fp32":
            progress_state["percent"] = 0.0
            progress_state["label"] = message
            progress_state["state"] = "running"
        elif event_type == "cancelled":
            progress_state["label"] = message or "PyMSS 分离任务已停止"
            progress_state["state"] = "stopped"
        elif event_type in {"fatal", "busy"}:
            progress_state["label"] = (
                message.splitlines()[0] if message else "PyMSS 分离任务失败"
            )
            progress_state["state"] = "failed"
            busy = event_type == "busy"
        elif event_type in {"preparing", "precision_attempt", "status"}:
            progress_state["label"] = message
            progress_state["state"] = "running"

    last_info = ""
    try:
        for info in _pymss_separate_core(
            model_name,
            inp_root,
            save_root_vocal,
            paths,
            save_root_ins,
            format0,
            event_callback=update_progress,
        ):
            last_info = info
            start_button = button_update()
            stop_button = button_update()
            if not busy:
                start_button = button_update(visible=False)
                stop_button = button_update(visible=True)
            yield (
                info,
                render_pymss_progress(**progress_state),
                start_button,
                stop_button,
            )
    except Exception:
        last_info = "失败\n%s" % traceback.format_exc()
        progress_state.update(
            {"label": "PyMSS 分离任务失败", "state": "failed"}
        )
        logger.exception("PyMSS WebUI task failed")

    start_button = button_update()
    stop_button = button_update()
    if not busy:
        start_button = button_update(visible=True)
        stop_button = button_update(visible=False)
    yield (last_info, render_pymss_progress(**progress_state), start_button, stop_button)


def stop_pymss_webui():
    return (
        _stop_pymss_separation_core(),
        render_pymss_progress(0, "PyMSS 分离任务已停止", "stopped"),
        button_update(visible=True),
        button_update(visible=False),
    )


def download_base_models_ui(keys, force):
    if not keys:
        raise gr.Error(i18n("请至少选择一个基础模型"))
    try:
        lines = download_base_assets(list(keys), force=bool(force))
    except Exception as error:
        logger.exception("Base model download failed")
        raise gr.Error(str(error))
    return "\n".join(lines)


def download_voice_model_ui(url, force):
    url = str(url or "").strip()
    if not url:
        raise gr.Error(i18n("请输入模型下载链接"))
    try:
        lines = download_voice_model_url(url, force=bool(force))
    except Exception as error:
        logger.exception("Voice model download failed")
        raise gr.Error(str(error))
    return "\n".join(lines), change_choices()


with gr.Blocks(title="RVC WebUI") as app:
    gr.Markdown("## RVC WebUI")
    gr.Markdown(
        value=i18n(
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
        )
    )
    with gr.Tabs():
        with gr.TabItem(i18n("模型推理")):
            with gr.Row():
                sid0 = gr.Dropdown(label=i18n("推理音色"), choices=sorted(names))
                with gr.Column():
                    refresh_button = gr.Button(
                        i18n("刷新音色列表"), variant="primary"
                    )
                    clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("请选择说话人id"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
                spk_item_dropdown = gr.Dropdown(
                    label=i18n("选择多说话人音色"),
                    choices=[],
                    value=None,
                    visible=False,
                    interactive=True,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            with gr.TabItem(i18n("单次推理")):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1, min_width=120):
                                    vc_transform0 = gr.Number(
                                        label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"),
                                        value=0,
                                    )
                                with gr.Column(scale=2, min_width=200):
                                    f0method0 = gr.Radio(
                                        label=i18n("选择音高提取算法"),
                                        choices=["pm", "rmvpe", "fcpe"],
                                        value="rmvpe",
                                        interactive=True,
                                    )
                            input_audio0 = gr.Audio(
                                label=i18n("拖拽或点击上传待处理音频"),
                                source="upload",
                                type="filepath",
                                interactive=True,
                            )

                        with gr.Column():
                            resample_sr0 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate0 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n(
                                    "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"
                                ),
                                value=0.25,
                                interactive=True,
                            )
                            protect0 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label=i18n(
                                    "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                                ),
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                            index_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("检索特征占比"),
                                value=0.75,
                                interactive=True,
                            )
                            file_index1 = gr.Textbox(
                                label=i18n("特征检索库文件路径（选择模型后自动匹配，可手动修改）"),
                                placeholder="C:\\Users\\Desktop\\model_example.index",
                                interactive=True,
                            )
                            refresh_button.click(
                                fn=change_choices,
                                inputs=[],
                                outputs=sid0,
                                api_name="infer_refresh",
                            )
                with gr.Group():
                    with gr.Column():
                        but0 = gr.Button(i18n("转换"), variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label=i18n("输出信息"))
                            vc_output2 = gr.Audio(
                                label=i18n("输出音频(右下角三个点,点了可以下载)")
                            )

                        but0.click(
                            report_missing_index,
                            [file_index1],
                            [],
                            queue=False,
                            api_name="infer_check_index",
                        )
                        but0.click(
                            vc_single_with_speaker,
                            [
                                spk_item,
                                spk_item_dropdown,
                                input_audio0,
                                vc_transform0,
                                f0method0,
                                file_index1,
                                index_rate1,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                            ],
                            [vc_output1, vc_output2],
                            api_name="infer_convert",
                        )
            with gr.TabItem(i18n("批量推理")):
                gr.Markdown(
                    value=i18n(
                        "批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. "
                    )
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"),
                            value=0,
                        )
                        opt_input = gr.Textbox(
                            label=i18n("指定输出文件夹"), value="opt"
                        )
                        file_index3 = gr.Textbox(
                            label=i18n("特征检索库文件路径（选择模型后自动匹配，可手动修改）"),
                            value="",
                            interactive=True,
                        )
                        f0method1 = gr.Radio(
                            label=i18n("选择音高提取算法"),
                            choices=["pm", "rmvpe", "fcpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        format1 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="wav",
                            interactive=True,
                        )

                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n(
                                "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"
                            ),
                            value=1,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=1,
                            interactive=True,
                        )
                with gr.Row():
                    dir_input = gr.Textbox(
                        label=i18n(
                            "输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"
                        ),
                        placeholder="C:\\Users\\Desktop\\input_vocal_dir",
                    )
                    inputs = gr.File(
                        file_count="multiple",
                        label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                    )

                with gr.Row():
                    but1 = gr.Button(i18n("转换"), variant="primary")
                    vc_output3 = gr.Textbox(label=i18n("输出信息"))

                    but1.click(
                        report_missing_index,
                        [file_index3],
                        [],
                        queue=False,
                        api_name="infer_check_index_batch",
                    )
                    but1.click(
                        vc_multi_with_speaker,
                        [
                            spk_item,
                            spk_item_dropdown,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            index_rate2,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                        ],
                        [vc_output3],
                        api_name="infer_convert_batch",
                    )
                sid0.change(
                    fn=vc.get_vc,
                    inputs=[sid0, protect0, protect1],
                    outputs=[
                        spk_item,
                        spk_item_dropdown,
                        protect0,
                        protect1,
                        file_index1,
                        file_index3,
                    ],
                    api_name="infer_change_voice",
                )
                spk_item.change(
                    fn=update_speaker_index,
                    inputs=[sid0, spk_item, spk_item_dropdown],
                    outputs=[file_index1, file_index3],
                    queue=False,
                    api_name="infer_change_speaker_index_slider",
                )
                spk_item_dropdown.change(
                    fn=update_dropdown_speaker_index,
                    inputs=[sid0, spk_item_dropdown],
                    outputs=[file_index1, file_index3],
                    queue=False,
                    api_name="infer_change_speaker_index_dropdown",
                )
        with gr.TabItem(i18n("人声伴奏分离&去混响")):
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "人声、伴奏与混响批量处理，使用pymss/MSST模型。"
                    )
                )
                with gr.Row():
                    with gr.Column():
                        dir_wav_input = gr.Textbox(
                            label=i18n("输入待处理音频文件夹路径"),
                            placeholder="C:\\Users\\Desktop\\todo-songs",
                        )
                        wav_inputs = gr.File(
                            file_count="multiple",
                            label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                        )
                    with gr.Column():
                        model_choose = gr.Dropdown(
                            label=i18n("处理方式"),
                            choices=pymss_names,
                            value=pymss_names[0],
                            interactive=True,
                        )
                        model_info = gr.Textbox(
                            label=i18n("底层模型"),
                            value=get_model_info(pymss_names[0]),
                            interactive=False,
                        )
                        model_choose.change(
                            get_model_info,
                            [model_choose],
                            [model_info],
                            queue=False,
                        )
                        opt_vocal_root = gr.Textbox(
                            label=i18n("主结果文件夹"), value="opt"
                        )
                        opt_ins_root = gr.Textbox(
                            label=i18n("分离残余文件夹"), value="opt"
                        )
                        format0 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                    with gr.Row():
                        but2 = gr.Button(i18n("转换"), variant="primary")
                        stop_pymss_button = gr.Button(
                            i18n("停止分离"), variant="stop", visible=False
                        )
                    pymss_progress = gr.HTML(
                        value=render_pymss_progress(0, "等待开始", "idle")
                    )
                    vc_output4 = gr.Textbox(label=i18n("输出信息"))
                    but2.click(
                        run_pymss_separation,
                        [
                            model_choose,
                            dir_wav_input,
                            opt_vocal_root,
                            wav_inputs,
                            opt_ins_root,
                            format0,
                        ],
                        [vc_output4, pymss_progress, but2, stop_pymss_button],
                        api_name="pymss_separate",
                    )
                    stop_pymss_button.click(
                        stop_pymss_webui,
                        [],
                        [vc_output4, pymss_progress, but2, stop_pymss_button],
                        queue=False,
                    )
        with gr.TabItem(i18n("模型下载")):
            download_info = gr.Textbox(
                label=i18n("输出信息"), value="", lines=6, max_lines=14
            )
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "下载推理所需基础模型（HuBERT 特征提取器与 RMVPE 音高检测），保存至 assets/ 目录。"
                    )
                )
                base_checks = gr.CheckboxGroup(
                    choices=sorted(BASE_ASSETS),
                    value=sorted(BASE_ASSETS),
                    label=i18n("基础模型"),
                )
                with gr.Row():
                    base_force = gr.Checkbox(
                        label=i18n("强制重新下载"), value=False
                    )
                    but_base = gr.Button(i18n("下载基础模型"), variant="primary")
                but_base.click(
                    download_base_models_ui,
                    [base_checks, base_force],
                    [download_info],
                    api_name="download_base_models",
                )
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "从 URL 下载音色模型：.pth 存入 assets/weights，.index 存入 assets/indices，zip 自动按扩展名分类。"
                    )
                )
                voice_url = gr.Textbox(
                    label="URL",
                    placeholder="https://example.com/model.pth",
                )
                with gr.Row():
                    voice_force = gr.Checkbox(
                        label=i18n("强制重新下载"), value=False
                    )
                    but_voice = gr.Button(i18n("下载音色模型"), variant="primary")
                but_voice.click(
                    download_voice_model_ui,
                    [voice_url, voice_force],
                    [download_info, sid0],
                    api_name="download_voice_model",
                )

        tab_faq = i18n("常见问题解答")
        with gr.TabItem(tab_faq):
            try:
                if tab_faq == "常见问题解答":
                    info = read_text("docs/cn/faq.md")
                else:
                    info = read_text("docs/en/faq_en.md")
                gr.Markdown(value=info)
            except Exception:
                gr.Markdown(traceback.format_exc())

    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        launch_webui_with_port_fallback(app, config)
