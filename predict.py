from ultralytics import YOLO

model = YOLO('./runs/detect/train8/weights/best.pt')

model.predict('./data/images/train/building workers_62.jpeg', save=True, show=True, conf=0.5 , save_txt=True)