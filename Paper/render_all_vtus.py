from paraview.simple import *
import math
import os
import glob

# ================= USER CONFIGURATION =================
# 指向刚才使用 pyvista 批量导出 vtu 文件的文件夹
INPUT_DIR = "Paper/vtu_outputs"
INPUT_DIR = "E:\_ResearchData\R2412VariableDensityTPMS\paper\out"

# 注意：刚才导出的网格数据中，标量名称为 "Material"
SCALAR_NAME = "data0"
MIN_THRESHOLD = 0.5  # Material 值为 1，0.5 即可完美过滤
RES = (2400, 2400)

# Morandi Palette (保留你的莫兰迪高级配色)
MORANDI_DARK = [0.34, 0.40, 0.49]
MORANDI_LIGHT = [0.85, 0.88, 0.92]
# ======================================================

# 获取目录下所有的 .vtu 文件
vtu_files = glob.glob(os.path.join(INPUT_DIR, "*.vtu"))

if not vtu_files:
    print(f"[!] 未在 {INPUT_DIR} 找到任何 .vtu 文件，请检查路径！")
    exit()

print(f"=== 找到 {len(vtu_files)} 个 .vtu 文件，开始批量 GPU 渲染 ===")

for file_path in vtu_files:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(INPUT_DIR, f"{base_name}_render.png")

    print(f"\n---> 正在处理: {base_name}")

    # 1. Load Data
    reader = XMLUnstructuredGridReader(FileName=[file_path])
    UpdatePipeline()
    print("     [-] Data loaded.")

    # 2. Thresholding
    threshold = Threshold(Input=reader)
    threshold.Scalars = ['CELLS', SCALAR_NAME]
    threshold.ThresholdMethod = 'Above Upper Threshold'
    threshold.UpperThreshold = MIN_THRESHOLD
    threshold.AllScalars = 1
    UpdatePipeline()
    print("     [-] Threshold applied.")

    # -------------------------------------------------------------
    # 【核心修复 1】：添加 ExtractSurface 提取外表面
    # 这一步能剔除百万级不可见的内部网格面，将 OptiX 显存占用降低 90% 以上！
    # -------------------------------------------------------------
    surface = ExtractSurface(Input=threshold)
    UpdatePipeline()
    print("     [-] Surface extracted (Optimization).")

    # 3. View Setup
    view = GetActiveViewOrCreate('RenderView')
    view.ViewSize = RES
    view.Background = [1.0, 1.0, 1.0]  # White
    view.OrientationAxesVisibility = 0
    view.AxesGrid.Visibility = 0

    # 4. GPU Engine (OptiX)
    view.EnableRayTracing = 1
    view.BackEnd = 'OSPRay pathtracer'

    # -------------------------------------------------------------
    # 【核心修复 2】：降低采样率
    # -------------------------------------------------------------
    view.SamplesPerPixel = 30
    view.Shadows = 0
    view.AmbientSamples = 16
    view.Denoise = 1

    try:
        view.ToneMappingType = 0
    except:
        pass

    # 5. Display & High-End Journal Colors (Step Rendering)
    display = Show(surface, view)
    display.SetRepresentationType('Surface')

    # --- 材质优化：模拟高级陶瓷/哑光树脂质感 ---
    display.Interpolation = 'PBR'
    display.Metallic = 0.05
    display.Roughness = 0.45
    display.Specular = 0.5

    ColorBy(display, ('CELLS', SCALAR_NAME))
    display.RescaleTransferFunctionToDataRange(True, False)

    lut = GetColorTransferFunction(SCALAR_NAME)

    # --- 开启阶梯渲染 ---
    lut.Discretize = 1
    lut.NumberOfTableValues = 6

    # --- 色彩搭配 ---
    lut.RGBPoints = [
        0.0, 0.17, 0.25, 0.38,
        0.2, 0.27, 0.46, 0.53,
        0.4, 0.55, 0.67, 0.65,
        0.6, 0.88, 0.82, 0.70,
        0.8, 0.80, 0.51, 0.36,
        1.0, 0.60, 0.20, 0.20
    ]
    lut.RescaleTransferFunction(0.0, 1.0)
    display.SetScalarBarVisibility(view, False)

    # 6. Lighting (Headlight)
    view.AdditionalLights = []
    light = CreateLight()
    light.Intensity = 1.5
    light.Type = 1  # Headlight
    light.DiffuseColor = [1.0, 1.0, 1.0]
    view.AdditionalLights = [light]

    # 7. Absolute Isometric Camera
    view.ResetCamera()
    camera = view.GetActiveCamera()
    bounds = surface.GetDataInformation().GetBounds()
    center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

    dist = max_dim * 2.8
    camera.SetFocalPoint(center)
    camera.SetPosition([center[0] + dist, center[1] - dist, center[2] + dist])
    camera.SetViewUp(0, 0, 1)
    view.ResetCamera()
    camera.Dolly(1.1)

    # 8. Render & Save
    print(f"     [*] Raytracing Engine Started... This might take a few seconds.")
    UpdatePipeline()
    Render()

    SaveScreenshot(output_file, view,
                   ImageResolution=RES,
                   TransparentBackground=0)

    print(f"     ✓ Done. Processed: {os.path.basename(output_file)}")

    # -------------------------------------------------------------
    # 【新增清理机制】：防止批量处理时管道堆积导致 GPU 显存溢出
    # -------------------------------------------------------------
    Delete(light)
    Delete(display)
    Delete(surface)
    Delete(threshold)
    Delete(reader)
    view.AdditionalLights = []

print("\n=== 所有渲染任务完成！请前往 vtu_outputs 文件夹查看高级渲染图 ===")