huaqiang_project/
├─ configs/
│  ├─ pc.yaml
│  └─ ascend.yaml
├─ src/
│  ├─ main_pc.py
│  ├─ main_310b.py
│  ├─ camera/
│  │  ├─ __init__.py
│  │  ├─ camera_reader.py
│  │  └─ camera_test.py
│  ├─ infer/
│  │  ├─ __init__.py
│  │  ├─ base_engine.py
│  │  ├─ onnx_engine.py
│  │  └─ ascend_engine.py
│  ├─ pipeline/
│  │  ├─ __init__.py
│  │  └─ runtime_pipeline.py
│  └─ utils/
│     ├─ __init__.py
│     └─ config.py
├─ models/
│  ├─ onnx/
│  └─ om/
├─ outputs/
│  ├─ logs/
│  └─ records/
└─ scripts/