import os
import gradio as gr
import yaml
from ultralytics import YOLO
import json

all_models_available = ['yolov3', 'yolov3-tiny', 'yolov3-spp',
                        'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
                        'yolov5-p6n', 'yolov5-p6s', 'yolov5-p6m', 'yolov5-p6l', 'yolov5-p6x',
                        'yolov6n', 'yolov6s', 'yolov6m', 'yolov6l', 'yolov6x',
                        'yolov7', 'yolov7-tiny', 'yolov7x',
                        'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
all_att_raw = [
    'IdenticalLayer',
    'CoordAtt',
    'CBAM_ATT',
    'ECAAttention',
    'SEAttention',
    'TripletAttention',
    'ShuffleAttention',
    'DoubleAttention', ]
with open("custom_layers.json", "r", encoding="utf-8") as f:
    custom_layer = json.load(f)
all_att = all_att_raw + list(custom_layer.keys())

iface1_inputs = [gr.Text(label="项目名称", info="项目名称(保存路径)", value="aaa"),
                 gr.Dropdown(choices=all_models_available, label="基础模型", info="请选择你需要的基础模型", value="yolov8n"),
                 gr.Dropdown(choices=all_att, label="机制1", info="第一个位置添加的机制", value='IdenticalLayer'),
                 gr.Dropdown(choices=all_att, label="机制2", info="第二个位置添加的机制", value='IdenticalLayer'),
                 gr.Dropdown(choices=all_att, label="机制3", info="第三个位置添加的机制", value='IdenticalLayer'),
                 gr.Dropdown(choices=all_att, label="机制4", info="第四个位置添加的机制", value='IdenticalLayer'),
                 gr.Dropdown(choices=all_att, label="机制5(如果有)", info="第五个位置添加的机制", value='IdenticalLayer'),
                 gr.Text(label="数据集", info="数据集路径", value=r"F:\studypython\yolov7-main\yolov7-main\data\tb1.yaml"),
                 gr.Text(label="图片大小AxB", info="图片大小，例如640", value="640x640"),
                 gr.Checkbox(label="是否GPU", value=True),
                 gr.Number(label="种子", info="没有则留空"),
                 gr.Number(label='workers数', info="workers数，一般为0", value=0),
                 gr.Slider(0, 500, value=1, step=1, label="epoch", info="训练的epoch数"),
                 gr.Slider(30, 80, value=50, step=1, label="patient", info="早停参数"),
                 gr.Slider(1, 128, value=8, step=1, label='batch大小', info="batch大小"), ]
iface1_outputs = [gr.Text(label='结果位置'),
                  gr.Text(label='基本结果'),
                  gr.Text(label='速度'),
                  gr.Image(label='结果图', type="filepath"),
                  gr.Image(label='混淆矩阵(归一化)', type="filepath"),
                  gr.Image(label='F1曲线', type="filepath"),
                  gr.Image(label='labels', type="filepath"),
                  gr.Image(label='labels相关矩阵', type="filepath"),
                  gr.Image(label='P曲线', type="filepath"),
                  gr.Image(label='PR曲线', type="filepath"),
                  gr.Image(label='R曲线', type="filepath"),
                  ]


def select_and_train(project_name, model_name, att1, att2, att3, att4, att5, data_yaml, img_size, is_gpu, random_seed,
                     workers, epoch,
                     patient, batch_size):
    while os.path.exists(os.path.join('runs', project_name)):
        if str.isdigit(project_name[-1]):
            project_name = project_name[:-1] + str(int(project_name[-1]) + 1)
        else:
            project_name = project_name + '1'
    os.mkdir(os.path.join('runs', project_name))
    big_version_name = model_name[4:6]
    is_raw = att1 == 'IdenticalLayer' and att2 == 'IdenticalLayer' and att3 == 'IdenticalLayer' \
             and att4 == 'IdenticalLayer' and att5 == 'IdenticalLayer'
    if big_version_name in ['v3', 'v7']:
        file_name = model_name + '.yaml'
    else:
        file_name = model_name[:-1] + '.yaml'
    if is_raw:
        yaml_path = os.path.join('ultralytics', 'cfg', 'models', big_version_name, file_name)
    else:
        yaml_path = os.path.join('ultralytics', 'cfg', 'models', 'custom', big_version_name, file_name)
    with open(yaml_path) as f:
        yaml_str = f.read()
    if not is_raw:
        if big_version_name in ['v3', 'v5', 'v7']:
            to_be_replaced = ['[-1, 1, IdenticalLayer,[]], # +1', '[-1, 1, IdenticalLayer,[]], # +2',
                              '[-1, 1, IdenticalLayer,[]], # +3', '[-1, 1, IdenticalLayer,[]], # +4',
                              '[-1, 1, IdenticalLayer,[]], # +5']
        else:
            to_be_replaced = ['[-1, 1, IdenticalLayer,[]] # +1', '[-1, 1, IdenticalLayer,[]] # +2',
                              '[-1, 1, IdenticalLayer,[]] # +3', '[-1, 1, IdenticalLayer,[]] # +4',
                              '[-1, 1, IdenticalLayer,[]] # +5']
        to_replace_str = [att1, att2, att3, att4, att5]
        to_replace = []
        if big_version_name in ['v3', 'v5', 'v7']:
            for index, model_str in enumerate(to_replace_str):
                if model_str in custom_layer.keys():
                    to_replace.append(custom_layer[model_str] + f', # +{index}')
                else:
                    to_replace.append(f'[-1, 1, {model_str},[]], # +{index}')
        else:
            for index, model_str in enumerate(to_replace_str):
                if model_str in custom_layer.keys():
                    to_replace.append(custom_layer[model_str] + f' # +{index}')
                else:
                    to_replace.append(f'[-1, 1, {model_str},[]] # +{index}')
        for i, j in zip(to_be_replaced, to_replace):
            yaml_str = yaml_str.replace(i, j)
    with open(data_yaml, 'r') as f:
        r = yaml.load(f.read(), Loader=yaml.FullLoader)
        if 'nc' in r.keys():
            nc = r['nc']
        else:
            nc = len(r['names'])
    yaml_str = yaml_str.replace('nc: 80  # number of classes', f'nc: {nc}  # number of classes')
    os.mkdir(os.path.join('runs', project_name, 'model'))
    with open(os.path.join('runs', project_name, 'model', model_name + '.yaml'), 'w') as f:
        f.write(yaml_str)
    with open(os.path.join('runs', project_name, 'model', 'data.yaml'), 'w') as f:
        yaml.dump(r, f)
    model = YOLO(os.path.join('runs', project_name, 'model', model_name + '.yaml'))
    imgsz = tuple([int(i) for i in img_size.split('x')])
    random_seed = random_seed if isinstance(random_seed, int) else 0

    results = model.train(data=data_yaml, epochs=epoch, patience=patient, imgsz=imgsz, batch=batch_size,
                          seed=random_seed, device='0' if is_gpu else 'cpu',
                          project=os.path.join('runs', project_name, 'result'), pretrained=False, workers=workers)
    return os.path.join('runs', project_name), results.results_dict, results.speed, \
           os.path.join('runs', project_name, 'result', 'train', 'results.png'), \
           os.path.join('runs', project_name, 'result', 'train', 'confusion_matrix_normalized.png'), \
           os.path.join('runs', project_name, 'result', 'train', 'F1_curve.png'), \
           os.path.join('runs', project_name, 'result', 'train', 'labels.jpg'), \
           os.path.join('runs', project_name, 'result', 'train', 'labels_correlogram.jpg'), \
           os.path.join('runs', project_name, 'result', 'train', 'P_curve.png'), \
           os.path.join('runs', project_name, 'result', 'train', 'PR_curve.png'), \
           os.path.join('runs', project_name, 'result', 'train', 'R_curve.png')


iface2_parallel_inputs = [gr.Text(label='自定义并联模块名称'),
                          gr.Checkboxgroup(label='并联模块', info='选择模块', choices=all_att)]
iface2_parallel_outputs = ['text']

iface2_cascade_inputs = [gr.Text(label='自定义串联模块名称'),
                         gr.Dropdown(choices=all_att, label="机制1", info="第1个位置串联的机制", value='IdenticalLayer'),
                         gr.Dropdown(choices=all_att, label="机制2", info="第2个位置串联的机制", value='IdenticalLayer'),
                         gr.Dropdown(choices=all_att, label="机制3", info="第3个位置串联的机制", value='IdenticalLayer'),
                         gr.Dropdown(choices=all_att, label="机制4", info="第4个位置串联的机制", value='IdenticalLayer'),
                         gr.Dropdown(choices=all_att, label="机制5", info="第5个位置串联的机制", value='IdenticalLayer'),
                         gr.Dropdown(choices=all_att, label="机制6", info="第6个位置串联的机制", value='IdenticalLayer'),
                         gr.Dropdown(choices=all_att, label="机制7", info="第7个位置串联的机制", value='IdenticalLayer'),
                         gr.Dropdown(choices=all_att, label="机制8", info="第8个位置串联的机制", value='IdenticalLayer'),
                         gr.Dropdown(choices=all_att, label="机制9", info="第9个位置串联的机制", value='IdenticalLayer'),
                         gr.Dropdown(choices=all_att, label="机制10", info="第9个位置串联的机制", value='IdenticalLayer'),
                         ]
iface2_cascade_outputs = ['text']


def add_parallel_module(module_name, module_list):
    result = [-1, 1, 'SuperLayerParallel', [module_list]]
    with open("custom_layers.json", "r", encoding="utf-8") as f:
        old_data = json.load(f)
        old_data.update({module_name: result.__str__()})
    with open("custom_layers.json", "w", encoding="utf-8") as f:
        json.dump(old_data, f)
    return 'success!'


def add_cascade_module(module_name, module1, module2, module3, module4, module5, module6, module7, module8, module9,
                       module10):
    module_list = [module1, module2, module3, module4, module5, module6, module7, module8, module9,
                   module10]
    result = [-1, 1, 'SuperLayerCascade', [module_list]]
    with open("custom_layers.json", "r", encoding="utf-8") as f:
        old_data = json.load(f)
        old_data.update({module_name: result.__str__()})
    with open("custom_layers.json", "w", encoding="utf-8") as f:
        json.dump(old_data, f)
    return 'success!'


iface3_inputs = [gr.Text(label='模型位置'),
                 gr.Slider(0, 1, value=0.3, label="iou_thresh", info="选择iou_thresh"),
                 gr.Slider(0, 1, value=0.7, label="conf_thresh", info="选择conf_thresh"),
                 gr.Image(label='输入图片', image_mode='RGB', type='filepath'),
                 ]
iface3_outputs = [gr.Text(label='预测结果'),
                  gr.Image(label='如图', type='filepath', image_mode='RGB')]


def view_model(model_path, iou_thresh, conf_thresh, the_img):
    model = YOLO(model_path)
    results = model.predict(the_img, iou=iou_thresh, conf=conf_thresh, save=True)
    return results[0].boxes, os.path.join(results[0].save_dir, results[0].path.split('\\')[-1])


iface1 = gr.Interface(inputs=iface1_inputs,
                      outputs=iface1_outputs,
                      fn=select_and_train,
                      description="如果出现什么问题，多重启几遍即可解决。模型改的过火可能会导致有个别结果图无法输出。。。")

iface2_parallel = gr.Interface(inputs=iface2_parallel_inputs,
                               outputs=iface2_parallel_outputs,
                               fn=add_parallel_module)
iface2_cascade = gr.Interface(inputs=iface2_cascade_inputs,
                              outputs=iface2_cascade_outputs,
                              fn=add_cascade_module)

iface2 = gr.TabbedInterface([iface2_parallel, iface2_cascade], ['定义并联模块', '定义串联模块'])
iface3 = gr.Interface(inputs=iface3_inputs,
                      outputs=iface3_outputs,
                      fn=view_model)

grapp = gr.TabbedInterface([iface1, iface2, iface3], ['模型选择和训练', '添加模块', '模型查看'],
                           title="YOLO调整平台v0.2")

if __name__ == "__main__":
    grapp.launch()
