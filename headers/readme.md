# Headers


## 1. config.h

Used it for configurate your custom model

To see shapes used [**netron**](https://netron.app/) - onnx viewer

I used yolov8n.engine, which includes x_center_bbox, y_center_bbox, width_bbox, height_bbox, confidence of classes [5:84].

To generate engine file you need get onnx file from ultralytics:

1. Install ultralytics 

`pip3 install ultralytics`

2. Generate onnx file 

If you want use default model, you can use my script `/supply_scripts/gen_onnx.py`

3. Find and run trtexec to generate engine file (**MUST BE INSTALLED TENSORRT AND CUDA-TOOLKIT**)

`trtexec --onnx=< path to model>/yolov8n.onnx --saveEngine=< outputpath to model>yolov8n.engine --fp16`

4. Open model in netron and change params


