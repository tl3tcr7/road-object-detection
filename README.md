# Road Object Detection

YOLOv8 model that detects 25 types of road objects, with Arabic/English text reading (EasyOCR), served through a Flask web app.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## Classes

Bike, Building, Car, Person, Stairs, Traffic sign, Electrical Pole, Road, Motorcycle, Dustbin, Dog, Manhole, Tree, Guard rail, Pedestrian crosswalk, Truck, Bus, Bench, Traffic Cone, Fire hydrant, Traffic Barrel, Plant Pot, Electrical Box, Chair, Bicycle Rack

## Structure

- `app.py`: Flask backend (YOLO + EasyOCR)
- `templates/`, `static/`: web interface
- `best.pt`: trained model weights
