# VisionLabSegmentationTask
Задача - семантическая сегментация органов брюшной полости.
Задача была решена для классов: селезенка, правая почка, левая почка, печень, желудок, аорта, поджелудочная железа

## Installation
```bash
conda create -n visionlab python=3.10
conda activate visionlab
# Установить pytorch с официального сайта https://pytorch.org
pip install -r requirements.txt
```

## Train
Цикл тренировки модели описывается с помощью конфигов в папке configs.
Перед тренировкой подготовьте датасет в формате images и labels и в поле конфига для датамодуля укажите путь
на корневую директорию датасета
```bash
# --configs - путь к конфигу тренировки
python train.py --config configs/base.yaml
```

## Predict
Для получения предсказаний можно либо взять инференс из predict.py, либо сразу воспользоваться этим скриптом
```bash
# --model - путь к весам модели
# --src - папка с *.npy файлами сканов
# --dst - папка куда будут загружены маски после инференса
python predict.py --model weights/best.pt --src dataset/images --dst dataset/predict
```